import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/utils/cn";
import {
  Adam,
  MOE_DEFAULT,
  Muon,
  Transformer,
  cosineLr,
  generate,
  mulberry32,
  trainStep,
  type Activation,
  type Optimizer,
} from "@/lib/ml";
import { buildTokenizer, corpora } from "@/lib/corpus";
import { STRINGS, SEEDS, LANGS, type Lang, type Seeds, type Strings } from "@/lib/i18n";
import { Gloss, GlossLang } from "@/components/Gloss";
import LossChart from "@/components/LossChart";
import Architecture from "@/components/Architecture";
import { Section, Card, Advanced, Utskrift } from "@/components/ui";
import Rlhf from "@/components/Rlhf";
import BpeLab from "@/components/BpeLab";
import Inspector from "@/components/Inspector";
import Skruer from "@/components/Skruer";
import Slankekur from "@/components/Slankekur";
import Leseliste from "@/components/Leseliste";
import Ordliste from "@/components/Ordliste";
import { EKSEMPELTEKSTER, type EksId } from "@/lib/eksempeltekster";
import Bekreft, { type Ask } from "@/components/Bekreft";
import { useRlhf } from "@/lib/useRlhf";
import { buildModelWorkbook } from "@/lib/excel-model";
import { buildModelGguf, downloadGguf } from "@/lib/gguf";
import { downloadWorkbook } from "@/lib/xlsx-zip";

const MAX_STEPS = 3500;
const CHUNK = 6;

type PresetKey = "liten" | "mellom" | "stor";

const PRESETS: Record<
  PresetKey,
  { dim: number; nLayer: number; nHead: number; seqLen: number; ffnMult: number }
> = {
  liten: { dim: 48, nLayer: 2, nHead: 2, seqLen: 32, ffnMult: 4 },
  mellom: { dim: 64, nLayer: 3, nHead: 2, seqLen: 40, ffnMult: 4 },
  stor: { dim: 96, nLayer: 4, nHead: 4, seqLen: 48, ffnMult: 4 },
};

type OptimKey = "adam" | "muon";

interface Engine {
  tokenizer: ReturnType<typeof buildTokenizer>;
  data: number[];
  model: Transformer;
  opt: Optimizer;
}

function charLabel(c: string): string {
  if (c === " ") return "␣";
  if (c === "\n") return "⏎";
  return c;
}

// Lærestripa: ekte utskrifter frå ei treningsøkt, skrivne fram teikn for teikn
// på linjert papir. Startteksten står dempa, og dei to første radene får
// lærarens raude bølgjestrek – berre under det modellen skreiv sjølv.
function LearningStrip({
  rows,
  seedText,
  t,
}: {
  rows: Seeds["strip"];
  seedText: string;
  t: Strings["hero"]["strip"];
}) {
  const total = useMemo(() => rows.reduce((n, r) => n + r.text.length, 0), [rows]);
  const [visible, setVisible] = useState(0);

  useEffect(() => {
    setVisible(0);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setVisible(total);
      return;
    }
    let i = 0;
    const id = window.setInterval(() => {
      i += 3;
      setVisible(i);
      if (i >= total) window.clearInterval(id);
    }, 24);
    return () => window.clearInterval(id);
  }, [total, rows]);

  let budget = visible;
  const shown = rows.map((r) => {
    const take = Math.max(0, Math.min(r.text.length, budget));
    budget -= take;
    return r.text.slice(0, take);
  });
  const done = visible >= total;

  return (
    <figure className="panel linjert relative mt-10 max-w-3xl overflow-hidden px-4 sm:px-6">
      <figcaption className="etikett" style={{ lineHeight: "2rem" }}>
        {t.title}
      </figcaption>
      {rows.map((r, i) => (
        <div key={r.step} className="flex items-baseline gap-3" style={{ lineHeight: "2rem" }}>
          <span className="w-16 flex-none text-right font-mono text-[10px] text-blyant sm:w-20">
            {t.step(r.step)}
          </span>
          <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-blekk">
            <Utskrift
              text={shown[i]}
              seed={seedText}
              restClassName={i < 2 ? "rettelinje" : undefined}
            />
          </span>
        </div>
      ))}
      <div className="text-right" style={{ lineHeight: "2rem" }}>
        <span
          aria-hidden
          className={`handnotat pr-2 text-xl transition-opacity duration-700 ${
            done ? "opacity-100" : "opacity-0"
          }`}
        >
          ✓ {t.note}
        </span>
      </div>
    </figure>
  );
}

const LANG_KEY = "trainer-lang";
function readStoredLang(): Lang {
  try {
    const v = localStorage.getItem(LANG_KEY);
    if (v === "bm" || v === "nn") return v;
  } catch {
    /* localStorage unavailable */
  }
  return "bm";
}
function writeStoredLang(l: Lang) {
  try {
    localStorage.setItem(LANG_KEY, l);
  } catch {
    /* ignore */
  }
}

export default function App() {
  // ---- language ----
  const [lang, setLang] = useState<Lang>(() => readStoredLang());
  const s = STRINGS[lang];
  const seed = SEEDS[lang];
  const activeCorpus = corpora[lang];
  const activeLocale = LANGS.find((l) => l.id === lang)!.locale;

  // ---- konfigurasjon ----
  const [preset, setPreset] = useState<PresetKey>("liten");
  const [batch, setBatch] = useState(4);
  const [lr, setLr] = useState(0.0008);
  const [optim, setOptim] = useState<OptimKey>("adam");
  const [act, setAct] = useState<Activation>("situ");
  const [moe, setMoe] = useState(false);
  const [schedule, setSchedule] = useState(false);
  const [extraText, setExtraText] = useState("");
  // Kva for klassikar-utdrag som står i tekstfeltet (steg 9). Blir nullstilt
  // så snart eleven redigerer, så kjeldelinja aldri lyg om innhaldet.
  const [sampleId, setSampleId] = useState<EksId | "">("");

  const cfg = useMemo(
    () => ({ ...PRESETS[preset], batch, lr }),
    [preset, batch, lr]
  );

  // ---- motor (modell + optimerar + data) ----
  const engineRef = useRef<Engine | null>(null);
  const runningRef = useRef(false);
  const stepRef = useRef(0);
  const lossesRef = useRef<number[]>([]);
  const rngRef = useRef(mulberry32(42));
  const sampleRngRef = useRef(mulberry32(7));
  const lrRef = useRef(lr);
  const scheduleRef = useRef(schedule);
  const timerRef = useRef<number | null>(null);
  const generateTimerRef = useRef<number | null>(null);
  const activeExtraTextRef = useRef("");
  // Treningsfart (steg per ms, glidande snitt) for «ca. X s igjen»
  const rateRef = useRef(0);
  const lastTickRef = useRef<{ t: number; step: number } | null>(null);

  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [currentSample, setCurrentSample] = useState("");
  const [paramCount, setParamCount] = useState(0);
  const [ask, setAsk] = useState<Ask | null>(null);
  // Tel opp for kvar ny motor, så skrue-visualiseringa veit når ho må nullstille utvalet sitt.
  const [engineGen, setEngineGen] = useState(0);
  // Har eleven vore innom slankekuren? Då tek rekneark-eksporten med arket om
  // 4 bit. Ein ny motor er ein ny start, så flagget følgjer engineGen.
  const [slankRan, setSlankRan] = useState(false);
  useEffect(() => setSlankRan(false), [engineGen]);

  const rlhf = useRlhf({
    getModel: () => engineRef.current?.model ?? null,
    getTokenizer: () => engineRef.current?.tokenizer ?? null,
    isTrained: () => stepRef.current > 0,
    baseRunning: running,
  });

  const buildEngine = useCallback((customText = activeExtraTextRef.current) => {
    activeExtraTextRef.current = customText;
    const fullText = activeCorpus + "\n" + customText;
    const tokenizer = buildTokenizer(fullText);
    const data = tokenizer.encode(fullText);
    const arch = PRESETS[preset];
    const model = new Transformer(
      {
        vocab: tokenizer.vocab,
        dim: arch.dim,
        nLayer: arch.nLayer,
        nHead: arch.nHead,
        seqLen: arch.seqLen,
        ffnMult: arch.ffnMult,
        act,
        moe: moe ? MOE_DEFAULT : undefined,
      },
      mulberry32(1337)
    );
    const opt: Optimizer =
      optim === "muon"
        ? new Muon(model.optimGroups(), lrRef.current)
        : new Adam(model.params, lrRef.current);
    engineRef.current = { tokenizer, data, model, opt };
    stepRef.current = 0;
    lossesRef.current = [];
    rngRef.current = mulberry32(42);
    sampleRngRef.current = mulberry32(7);
    setStep(0);
    setLosses([]);
    setCurrentSample("");
    setParamCount(model.paramCount());
    setEngineGen((g) => g + 1);
    rlhf.reset();
    // arkitektur (preset, aktivering, ekspertar) og val av optimerar tvingar
    // fram ein ny modell – ikkje lr/batch, som kan endrast midt i ei økt
  }, [preset, act, moe, optim, rlhf.reset, activeCorpus]);

  useEffect(() => {
    if (!runningRef.current) buildEngine();
  }, [buildEngine]);

  // Endra læringsrate utan å byggja modellen på nytt (mistar ikkje framdrift).
  // Med nedtrapping på styrer treningsløkka raten steg for steg; når ho blir
  // slått av, skal den flate raten gjelda med ein gong.
  useEffect(() => {
    lrRef.current = lr;
    scheduleRef.current = schedule;
    if (engineRef.current && !schedule) engineRef.current.opt.lr = lr;
  }, [lr, schedule]);

  // ---- treningsløkke ----
  const loop = useCallback(() => {
    timerRef.current = null;
    if (!runningRef.current) return;
    const eng = engineRef.current;
    if (!eng) return;
    const seqLen = eng.model.seqLen;
    const stepsThisChunk = Math.min(CHUNK, MAX_STEPS - stepRef.current);
    for (let i = 0; i < stepsThisChunk; i++) {
      if (scheduleRef.current)
        eng.opt.lr = cosineLr(stepRef.current, { peak: lrRef.current, total: MAX_STEPS });
      const l = trainStep(eng.model, eng.opt, eng.data, seqLen, cfg.batch, rngRef.current);
      lossesRef.current.push(l);
      stepRef.current++;
    }
    if (lossesRef.current.length > MAX_STEPS)
      lossesRef.current = lossesRef.current.slice(-MAX_STEPS);
    const now = performance.now();
    const prev = lastTickRef.current;
    if (prev && stepRef.current > prev.step) {
      const inst = (stepRef.current - prev.step) / Math.max(1, now - prev.t);
      rateRef.current = rateRef.current ? rateRef.current * 0.8 + inst * 0.2 : inst;
    }
    lastTickRef.current = { t: now, step: stepRef.current };
    setStep(stepRef.current);
    setLosses(lossesRef.current.slice());
    if (stepRef.current % 60 < stepsThisChunk) {
      const s = generate(
        eng.model,
        eng.tokenizer.decode,
        eng.tokenizer.encode,
        seed.trainSeed,
        { temperature: 0.8, topK: 8, length: 90 },
        sampleRngRef.current
      );
      setCurrentSample(s);
    }
    if (stepRef.current >= MAX_STEPS) {
      runningRef.current = false;
      setRunning(false);
      return;
    }
    timerRef.current = window.setTimeout(loop, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg.batch, seed.trainSeed]);

  // Alt som kastar bort trening går gjennom denne: har eleven ikkje trent noko
  // enno, er det ingenting å mista, og då spør vi ikkje.
  const guard = useCallback(
    (whatKey: keyof Strings["confirm"]["what"], run: () => void) => {
      if (stepRef.current === 0) {
        run();
        return;
      }
      setAsk({
        title: s.confirm.title,
        what: s.confirm.what[whatKey],
        note: s.confirm.steps(stepRef.current.toLocaleString(activeLocale)),
        body: s.confirm.body,
        yes: s.confirm.yes,
        no: s.confirm.no,
        onYes: run,
      });
    },
    [s.confirm, activeLocale]
  );

  const beginTraining = useCallback(() => {
    rlhf.reset();
    if (!engineRef.current || stepRef.current >= MAX_STEPS) buildEngine();
    lastTickRef.current = null; // ikkje la pausetid forureine farten
    runningRef.current = true;
    setRunning(true);
    loop();
  }, [buildEngine, loop, rlhf.reset]);

  // Å halda fram ei pausa økt er trygt. Å trykkja start når den førre økta er
  // ferdig byggjer modellen på nytt – då må vi spørja først.
  const start = useCallback(() => {
    if (engineRef.current && stepRef.current >= MAX_STEPS) {
      guard("restart", beginTraining);
      return;
    }
    beginTraining();
  }, [beginTraining, guard]);

  // Verkstedknappen i heroen: start treninga med ein gong og hopp til oppgåve 5,
  // så modellen lærer i bakgrunnen medan ein pratar seg gjennom oppgåve 1–2.
  const startFromHero = useCallback(() => {
    start();
    const behavior: ScrollBehavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto"
      : "smooth";
    document.getElementById("trening")?.scrollIntoView({ behavior, block: "start" });
  }, [start]);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    stop();
    buildEngine();
  }, [stop, buildEngine]);

  // ---- last ned modellen som rekneark ----
  // Ingen makroar, ingen VBA – arket gjer inferens med vanlege formlar.
  // Sjå src/lib/excel-model.ts. Knappen ligg i botnteksten, og berre når det
  // finst noko trent (ei tilfeldig modell ville berre skrive støy).
  const [eggState, setEggState] = useState<"idle" | "working" | "done">("idle");
  const onSlankRan = useCallback(() => setSlankRan(true), []);

  const onExcelClick = useCallback(() => {
    if (eggState === "working") return;
    const eng = engineRef.current;
    if (!eng || stepRef.current === 0) return;

    setEggState("working");
    // Gje nettlesaren ein frame til å teikna «working» før vi blokkerer tråden.
    window.setTimeout(async () => {
      try {
        const built = buildModelWorkbook({
          model: eng.model,
          tokenizer: eng.tokenizer,
          prompt: seed.trainSeed,
          nGen: 16,
          step: stepRef.current,
          loss: lossesRef.current.length ? lossesRef.current[lossesRef.current.length - 1] : 0,
          presetName: preset,
          lang,
          includeQuant: slankRan,
        });
        await downloadWorkbook(
          built.workbook,
          `sprakmodell-${preset}-steg${stepRef.current}.xlsx`
        );
        setEggState("done");
        window.setTimeout(() => setEggState("idle"), 6000);
      } catch (err) {
        console.error("regneark-eksport feila", err);
        setEggState("idle");
      }
    }, 32);
  }, [eggState, lang, preset, seed.trainSeed, slankRan]);

  // ---- last ned modellen som GGUF ----
  // Same filformatet som Llama og Mistral blir delte i. Sjå src/lib/gguf.ts for
  // kva vi kan og ikkje kan love om henne.
  const [ggufState, setGgufState] = useState<"idle" | "working" | "done">("idle");

  const onGgufClick = useCallback(() => {
    if (ggufState === "working") return;
    const eng = engineRef.current;
    if (!eng || stepRef.current === 0) return;

    setGgufState("working");
    window.setTimeout(async () => {
      try {
        const built = buildModelGguf({
          model: eng.model,
          tokenizer: eng.tokenizer,
          step: stepRef.current,
          loss: lossesRef.current.length ? lossesRef.current[lossesRef.current.length - 1] : 0,
          presetName: preset,
          lang,
        });
        await downloadGguf(built.bytes, `sprakmodell-${preset}-steg${stepRef.current}.gguf`);
        setGgufState("done");
        window.setTimeout(() => setGgufState("idle"), 6000);
      } catch (err) {
        console.error("gguf-eksport feila", err);
        setGgufState("idle");
      }
    }, 32);
  }, [ggufState, lang, preset]);

  // Nullstilling kastar ei trena økt – krev to trykk når det finst noko å miste.
  const onResetClick = useCallback(() => guard("reset", reset), [guard, reset]);

  const rebuildWithExtraText = useCallback(
    () =>
      guard("text", () => {
        stop();
        buildEngine(extraText);
      }),
    [guard, stop, buildEngine, extraText]
  );

  // Finpussinga rullar vektene tilbake til referansemodellen, så ho har si eiga
  // rute: sjølve treninga overlever, men vala til brukaren gjer det ikkje.
  const onResetTuning = useCallback(() => {
    if (rlhf.metrics.count === 0) {
      rlhf.resetTuning();
      return;
    }
    setAsk({
      title: s.confirm.tuning.title,
      what: s.confirm.tuning.what,
      body: s.confirm.tuning.body,
      yes: s.confirm.tuning.yes,
      no: s.confirm.no,
      onYes: rlhf.resetTuning,
    });
  }, [rlhf.metrics.count, rlhf.resetTuning, s.confirm]);

  useEffect(
    () => () => {
      runningRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      if (generateTimerRef.current !== null) window.clearTimeout(generateTimerRef.current);
    },
    []
  );

  // ---- generering / "chat" ----
  const [chatPrompt, setChatPrompt] = useState(seed.chatPrompt);
  // startteksten som faktisk vart brukt: feltet over kan redigerast medan svaret
  // står, og då skal markeringa i svaret ikkje flytta seg
  const [chatSeed, setChatSeed] = useState("");
  const [chatFull, setChatFull] = useState("");
  const [chatShown, setChatShown] = useState("");
  const [genTemp, setGenTemp] = useState(0.7);
  const [genTopK, setGenTopK] = useState(8);
  const [genLen, setGenLen] = useState(120);
  const [genLoading, setGenLoading] = useState(false);
  // aukar for kvar generering, så skriveffekten startar på nytt sjølv om teksten er lik
  const [genTick, setGenTick] = useState(0);

  const runGenerate = useCallback(() => {
    const eng = engineRef.current;
    if (!eng) return;
    setGenLoading(true);
    setChatShown("");
    if (generateTimerRef.current !== null) window.clearTimeout(generateTimerRef.current);
    // la nettlesaren måle oppdateringa før tungt arbeid
    generateTimerRef.current = window.setTimeout(() => {
      generateTimerRef.current = null;
      if (engineRef.current !== eng) {
        setGenLoading(false);
        return;
      }
      const out = generate(
        eng.model,
        eng.tokenizer.decode,
        eng.tokenizer.encode,
        chatPrompt,
        { temperature: genTemp, topK: genTopK, length: genLen },
        sampleRngRef.current
      );
      setChatFull(out);
      setChatSeed(chatPrompt);
      setGenTick((t) => t + 1);
      setGenLoading(false);
    }, 20);
  }, [chatPrompt, genTemp, genTopK, genLen]);

  // progressiv avsløring av teksten (skriveffekt)
  useEffect(() => {
    if (!chatFull) return;
    let i = 0;
    setChatShown("");
    const id = window.setInterval(() => {
      i += 2;
      setChatShown(chatFull.slice(0, i));
      if (i >= chatFull.length) window.clearInterval(id);
    }, 16);
    return () => window.clearInterval(id);
  }, [genTick, chatFull]);

  // ---- tokeniserings-framsyning ----
  const [showFullCorpus, setShowFullCorpus] = useState(false);
  const displayTok = useMemo(() => buildTokenizer(activeCorpus), [activeCorpus]);
  const sampleSentence = seed.sampleSentence;
  const sampleTokens = useMemo(
    () => Array.from(sampleSentence).map((c) => ({ c, id: displayTok.stoi[c] })),
    [displayTok, sampleSentence]
  );
  const vocabList = useMemo(() => displayTok.itos, [displayTok]);

  const stats = useMemo(() => {
    const eng = engineRef.current;
    return {
      params: paramCount,
      vocab: eng?.tokenizer.vocab ?? displayTok.vocab,
      chars: eng?.data.length ?? activeCorpus.length,
      last: losses.length ? losses[losses.length - 1] : 0,
    };
  }, [paramCount, losses, displayTok, activeCorpus]);

  const examples = seed.examples;

  const getParams = useCallback(() => engineRef.current?.model.params ?? null, []);
  const getEngine = useCallback(() => engineRef.current, []);

  const trainedDone = step >= MAX_STEPS && !running;
  const eta = (() => {
    if (!running || !rateRef.current || step >= MAX_STEPS) return null;
    const sec = (MAX_STEPS - step) / rateRef.current / 1000;
    if (sec < 3) return null;
    return sec >= 90
      ? s.train.etaMin(Math.ceil(sec / 60))
      : s.train.etaSec(Math.max(5, Math.round(sec / 5) * 5));
  })();

  // Persist + reflect language on <html> and title.
  useEffect(() => {
    writeStoredLang(lang);
    const meta = LANGS.find((l) => l.id === lang)!;
    document.documentElement.lang = meta.htmlLang;
    document.title = s.docTitle;
  }, [lang, s.docTitle]);

  // On language change, repopulate the editable prompts with the new seed.
  useEffect(() => {
    setChatPrompt(seed.chatPrompt);
    rlhf.setPrompt(seed.chatPrompt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  return (
    <GlossLang.Provider value={lang}>
    <div className="min-h-screen">
      <Bekreft ask={ask} onClose={() => setAsk(null)} />

      {/* Header */}
      <header className="sticky top-0 z-20 border-b-2 border-blekk bg-papir/95 backdrop-blur">
        <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-x-3 gap-y-2 px-3 py-2.5 sm:px-4">
          <div className="flex items-center gap-2.5">
            <div className="hidden h-9 w-9 flex-none items-center justify-center rounded-[3px] bg-blekk pt-0.5 font-display text-sm leading-none text-white sm:flex">
              Aa
            </div>
            <div className="leading-tight">
              <div className="font-display text-[13px] text-blekk">{s.header.title}</div>
              <div className="hidden font-mono text-[10px] text-blyant sm:block">{s.header.subtitle}</div>
            </div>
            <div className="ml-2 inline-flex overflow-hidden rounded-[3px] border-2 border-blekk font-mono text-xs font-semibold">
              {LANGS.map((l) => (
                <button
                  key={l.id}
                  // Å trykkja på språket ein alt står i skal ikkje spørja om noko.
                  onClick={() => {
                    if (l.id !== lang) guard("lang", () => setLang(l.id));
                  }}
                  disabled={running || rlhf.dpoRunning}
                  className={cn(
                    "px-2 py-1 transition disabled:opacity-50 sm:px-2.5",
                    lang === l.id ? "bg-blekk text-white" : "bg-white text-blekk hover:bg-papir"
                  )}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>
          <div className="ml-auto flex flex-none items-center gap-3">
            {/* Levande status: gruppa kan vandre på sida medan eleven øver */}
            {(running || step > 0) && (
              <div className="hidden items-center gap-2 font-mono text-[11px] sm:flex">
                <span
                  className={cn(
                    "h-2 w-2 rounded-full",
                    running ? "animate-pulse bg-rettepenn" : "bg-blyant/40"
                  )}
                />
                <span className="text-blyant">{s.train.step(step, MAX_STEPS)}</span>
                {losses.length > 0 && (
                  <span className="font-semibold text-rettepenn">
                    {s.loss.axisLoss} {stats.last.toFixed(2)}
                  </span>
                )}
              </div>
            )}
            <a href="#trening" className="knapp knapp-blekk knapp-sm">
              {s.header.jump}
            </a>
          </div>
        </div>
      </header>

      {/* Omslaget: tittel med tusj-strek + lærestripa */}
      <div className="mx-auto max-w-4xl pr-4" style={{ paddingLeft: "var(--gutter)" }}>
        <div className="pb-12 pt-10 sm:pt-14">
          <div className="etikett mb-5 flex items-center gap-2">
            <span className="h-2 w-2 flex-none animate-pulse rounded-full bg-rettepenn" />
            {s.hero.badge}
          </div>
          <h1 className="max-w-3xl font-display text-[2.35rem] leading-[1.18] text-blekk sm:text-[3.25rem] sm:leading-[1.14]">
            {s.hero.h1Pre} <span className="tusj-strek">{s.hero.h1Lang}</span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-blyant">
            <Gloss text={s.hero.para} />
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <button
              onClick={startFromHero}
              disabled={running || rlhf.dpoRunning}
              className="knapp knapp-blekk"
            >
              {s.hero.ctaStart}
            </button>
            <a href="#forsta" className="knapp knapp-omriss">
              {s.hero.ctaUnderstand}
            </a>
          </div>
          <p className="mt-6 flex flex-wrap gap-x-2 gap-y-1 font-mono text-xs text-blyant">
            {s.hero.stats.map((st, i) => (
              <span key={st.v}>
                {i > 0 && <span className="mr-2 text-marg">·</span>}
                <b className="font-semibold text-blekk">{st.k}</b> {st.v}
              </span>
            ))}
          </p>
          <LearningStrip rows={seed.strip} seedText={seed.trainSeed} t={s.hero.strip} />
          <p className="mt-3 max-w-3xl font-mono text-[11px] leading-relaxed text-blyant">
            {s.hero.strip.caption} {s.seedLegend}
          </p>
        </div>
      </div>

      <main className="side mx-auto max-w-4xl space-y-16 pb-16 pr-4 pt-6">
        {/* Forstå */}
        <Section
          id="forsta"
          step={1}
          title={s.understand.title}
          intro={s.understand.intro}
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {s.understand.cards.map((c, i) => (
              <Card key={i}>
                <div aria-hidden className="font-display text-2xl text-blekk">
                  {i + 1}.
                </div>
                <div className="mt-2 font-semibold text-blekk">{c.t}</div>
                <p className="mt-1 text-sm leading-relaxed text-blyant">
                  <Gloss text={c.d} />
                </p>
              </Card>
            ))}
          </div>
        </Section>

        {/* Data & tokenisering */}
        <Section
          id="data"
          step={2}
          title={s.data.title}
          intro={s.data.intro}
        >
          <Card className="space-y-5">
            <div>
              <div className="mb-2 flex items-center justify-between gap-3">
                <h3 className="font-semibold text-blekk">{s.data.snippetHeading}</h3>
                <span className="font-mono text-xs text-blyant">
                  {s.data.charsTotal(stats.chars)}
                </span>
              </div>
              {/* lesestykket: teksten eleven skal lære av, satt som i ei lesebok */}
              <pre
                className={cn(
                  "overflow-auto whitespace-pre-wrap rounded-[2px] border border-blekk/25 bg-papir/70 p-4 font-sans text-[15px] leading-relaxed text-blekk",
                  !showFullCorpus && "max-h-40"
                )}
              >
{showFullCorpus ? activeCorpus : `${activeCorpus.slice(0, 420)}…`}
              </pre>
              <button
                onClick={() => setShowFullCorpus((v) => !v)}
                aria-expanded={showFullCorpus}
                className="mt-2 font-mono text-xs font-semibold text-blekk"
              >
                {showFullCorpus ? s.data.showLess : s.data.showFull}
              </button>
            </div>

            <div>
              <h3 className="mb-2 font-semibold text-blekk">{s.data.originHeading}</h3>
              <p className="max-w-2xl text-sm leading-relaxed text-blyant">
                <Gloss text={s.data.originPara} />
              </p>
            </div>

            <div>
              <h3 className="mb-2 font-semibold text-blekk">{s.data.howHeading}</h3>
              <p className="mb-3 text-sm text-blyant">
                {s.data.howPara(sampleSentence)}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {sampleTokens.map((t, i) => (
                  <span key={i} className="brikke flex-col px-2 py-1 text-center">
                    <span className="font-mono text-base font-semibold text-blekk">
                      {charLabel(t.c)}
                    </span>
                    <span className="font-mono text-[10px] text-blyant">#{t.id}</span>
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-2 font-semibold text-blekk">
                {s.data.vocabHeading(vocabList.length)}
              </h3>
              <div className="flex flex-wrap gap-1">
                {vocabList.map((c, i) => (
                  <span
                    key={i}
                    title={s.data.charTooltip(i)}
                    className="brikke h-7 min-w-7 px-1.5 text-sm text-blekk"
                  >
                    {charLabel(c)}
                  </span>
                ))}
              </div>
            </div>
          </Card>
        </Section>

        {/* Fra tegn til ord-biter (BPE) */}
        <Section
          id="bpe"
          step={3}
          title={s.bpe.title}
          intro={s.bpe.intro}
          fold={s.fold}
        >
          <Card>
            <BpeLab corpus={activeCorpus} sampleSentence={sampleSentence} s={s.bpe} />
          </Card>
        </Section>

        {/* Arkitektur */}
        <Section
          id="arkitektur"
          step={4}
          title={s.arch.title}
          intro={s.arch.intro}
          fold={s.fold}
        >
          <Card>
            <Architecture
              layers={cfg.nLayer}
              heads={cfg.nHead}
              dim={cfg.dim}
              moe={moe ? MOE_DEFAULT : undefined}
              s={s}
            />
            <div className="mt-5 grid grid-cols-1 gap-3 text-sm leading-relaxed text-blyant sm:grid-cols-2">
              <div className="rounded-[2px] border border-blekk/25 bg-papir/70 p-3">
                <b className="text-blekk">{s.arch.causalTitle}</b> <Gloss text={s.arch.causalBody} />
              </div>
              <div className="rounded-[2px] border border-blekk/25 bg-papir/70 p-3">
                <b className="text-blekk">{s.arch.headsTitle}</b> <Gloss text={s.arch.headsBody} />
              </div>
            </div>
          </Card>
        </Section>

        {/* Trening */}
        <Section
          id="trening"
          step={5}
          title={s.train.title}
          intro={s.train.intro}
        >
          <Card className="space-y-5">
            {/* kontrollar: berre modellstorleik synleg – resten er fordjuping */}
            <div className="max-w-sm">
              <label className="etikett mb-1 block">{s.train.modelSize}</label>
              <select
                value={preset}
                disabled={running || rlhf.dpoRunning}
                onChange={(e) => {
                  const v = e.target.value as PresetKey;
                  guard("preset", () => setPreset(v));
                }}
                className="felt text-sm disabled:opacity-50"
              >
                {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
                  <option key={k} value={k}>
                    {s.train.presets[k]}
                  </option>
                ))}
              </select>
            </div>

            <Advanced label={s.advanced}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label className="etikett mb-1 block"><Gloss text={s.train.minibatch(cfg.batch)} /></label>
                  <input
                    type="range"
                    min={1}
                    max={8}
                    value={batch}
                    disabled={running || rlhf.dpoRunning}
                    onChange={(e) => setBatch(Number(e.target.value))}
                    className="w-full disabled:opacity-50"
                  />
                </div>
                <div>
                  <label className="etikett mb-1 block"><Gloss text={s.train.learningRate(lr.toFixed(4))} /></label>
                  <input
                    type="range"
                    min={0.0001}
                    max={0.003}
                    step={0.0001}
                    value={lr}
                    disabled={running || rlhf.dpoRunning}
                    onChange={(e) => setLr(Number(e.target.value))}
                    className="w-full disabled:opacity-50"
                  />
                </div>
                <div>
                  <label className="etikett mb-1 block" htmlFor="optim">
                    {s.train.optimLabel}
                  </label>
                  <select
                    id="optim"
                    value={optim}
                    disabled={running || rlhf.dpoRunning}
                    onChange={(e) => {
                      const v = e.target.value as OptimKey;
                      guard("optim", () => setOptim(v));
                    }}
                    className="felt text-sm disabled:opacity-50"
                  >
                    <option value="adam">{s.train.optimAdam}</option>
                    <option value="muon">{s.train.optimMuon}</option>
                  </select>
                  <p className="mt-1 text-xs leading-relaxed text-blyant"><Gloss text={s.train.optimHelp} /></p>
                </div>
                <div>
                  <label className="etikett mb-1 block" htmlFor="akt">
                    {s.train.actLabel}
                  </label>
                  <select
                    id="akt"
                    value={act}
                    disabled={running || rlhf.dpoRunning}
                    onChange={(e) => {
                      const v = e.target.value as Activation;
                      guard("act", () => setAct(v));
                    }}
                    className="felt text-sm disabled:opacity-50"
                  >
                    <option value="gelu">{s.train.actGelu}</option>
                    <option value="situ">{s.train.actSitu}</option>
                  </select>
                  <p className="mt-1 text-xs leading-relaxed text-blyant"><Gloss text={s.train.actHelp} /></p>
                </div>
                <div>
                  <label className="etikett mb-1 block" htmlFor="moe">
                    {s.train.moeLabel}
                  </label>
                  <select
                    id="moe"
                    value={moe ? "on" : "off"}
                    disabled={running || rlhf.dpoRunning}
                    onChange={(e) => {
                      const v = e.target.value === "on";
                      guard("moe", () => setMoe(v));
                    }}
                    className="felt text-sm disabled:opacity-50"
                  >
                    <option value="off">{s.train.moeOff}</option>
                    <option value="on">{s.train.moeOn}</option>
                  </select>
                  <p className="mt-1 text-xs leading-relaxed text-blyant"><Gloss text={s.train.moeHelp} /></p>
                </div>
                <div className="sm:col-span-2">
                  <label className="flex items-start gap-2 text-sm text-blekk">
                    <input
                      type="checkbox"
                      checked={schedule}
                      onChange={(e) => setSchedule(e.target.checked)}
                      className="mt-0.5"
                    />
                    <span>
                      {s.train.scheduleLabel}
                      <span className="mt-1 block text-xs leading-relaxed text-blyant">
                        <Gloss text={s.train.scheduleHelp} />
                      </span>
                    </span>
                  </label>
                </div>
              </div>
            </Advanced>

            {/* knappar */}
            <div className="flex flex-wrap items-center gap-3">
              {!running ? (
                <button onClick={start} disabled={rlhf.dpoRunning} className="knapp knapp-blekk">
                  {s.train.start}
                </button>
              ) : (
                <button onClick={stop} className="knapp knapp-rettepenn">
                  {s.train.stop}
                </button>
              )}
              <button
                onClick={onResetClick}
                disabled={running}
                className="knapp knapp-omriss"
              >
                {s.train.reset}
              </button>
              {trainedDone && (
                <span aria-hidden className="stempel">
                  {s.train.stamp}
                </span>
              )}
              <div className="ml-auto text-right">
                <div className="font-mono text-sm font-semibold text-blekk">
                  {s.train.step(step, MAX_STEPS)}
                </div>
                <div className="font-mono text-xs text-blyant">{stats.params.toLocaleString(activeLocale)} <Gloss text={s.train.params} /></div>
                {eta && <div className="font-mono text-[11px] text-blyant">{eta}</div>}
              </div>
            </div>

            {/* framdrift */}
            <div className="h-2.5 w-full overflow-hidden rounded-[2px] border border-blekk/30 bg-white">
              <div
                className="h-full bg-blekk transition-all"
                style={{ width: `${(step / MAX_STEPS) * 100}%` }}
              />
            </div>

            {/* liten trøstemelding når maskina byrjar å jobbe */}
            {running && (
              <p className="text-xs leading-relaxed text-blyant" role="status">
                {s.train.fanNote}
              </p>
            )}

            {/* tap-graf */}
            <div>
              <h3 className="mb-2 font-semibold text-blekk">{s.train.lossHeading}</h3>
              <LossChart data={losses} loss={s.loss} />
              <p className="mt-2 text-xs leading-relaxed text-blyant">
                <Gloss text={s.train.lossHelp} />
              </p>
            </div>

            {/* fordypning: sjå vektene («skruane») bli vridde i sanntid */}
            <Advanced label={s.train.screwsLabel}>
              <Skruer
                getParams={getParams}
                step={step}
                engineGen={engineGen}
                lr={lr}
                help={s.train.screwsHelp}
                idleText={s.train.screwsIdle}
              />
            </Advanced>

            {/* fordypning: same krymping som dei store modellane gjer før drift */}
            <Advanced label={s.train.slank.label}>
              <Slankekur
                getEngine={getEngine}
                step={step}
                engineGen={engineGen}
                prompt={seed.trainSeed}
                locale={activeLocale}
                s={s.train.slank}
                onRan={onSlankRan}
              />
            </Advanced>

            {/* live-eksempel: eleven skriv på tavla */}
            <div className="tavle p-4">
              <div className="mb-2 flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-kritt/70">
                <span className={cn("h-2 w-2 rounded-full", running ? "animate-pulse bg-tusj" : "bg-kritt/30")} />
                {s.train.liveLabel}
              </div>
              <p className="min-h-6 whitespace-pre-wrap font-mono text-sm text-kritt">
                {currentSample ? (
                  <Utskrift text={currentSample} seed={seed.trainSeed} />
                ) : (
                  <span className="text-kritt/50">{s.train.livePlaceholder}</span>
                )}
              </p>
              {currentSample && (
                <p className="mt-2 text-[11px] leading-relaxed text-kritt/55">{s.seedLegend}</p>
              )}
            </div>
          </Card>
        </Section>

        {/* Se inni modellen */}
        <Section
          id="inspect"
          step={6}
          title={s.inspect.title}
          intro={s.inspect.intro}
        >
          <Card>
            <Inspector
              model={engineRef.current?.model ?? null}
              tokenizer={engineRef.current?.tokenizer ?? null}
              step={step}
              defaultText={seed.sampleSentence}
              s={s.inspect}
            />
          </Card>
        </Section>

        {/* Chat / generering */}
        <Section
          id="chat"
          step={7}
          title={s.chat.title}
          intro={s.chat.intro}
        >
          <Card className="space-y-5">
            <div>
              <label className="etikett mb-1 block">{s.chat.promptLabel}</label>
              <textarea
                value={chatPrompt}
                onChange={(e) => setChatPrompt(e.target.value)}
                rows={2}
                className="felt resize-none"
                placeholder={s.chat.promptPlaceholder}
              />
              <div className="mt-2 flex flex-wrap gap-2">
                {examples.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => setChatPrompt(ex)}
                    className="rounded-[2px] border border-blekk/40 bg-white px-3 py-1 font-mono text-xs text-blekk transition hover:bg-papir"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            {/* temperatur er den eine morosame brytaren – resten er fordjuping */}
            <div className="max-w-sm">
              <label className="etikett mb-1 block"><Gloss text={s.chat.temp(genTemp.toFixed(2))} /></label>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={genTemp}
                onChange={(e) => setGenTemp(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-[11px] text-blyant">{s.chat.tempHelp}</p>
            </div>

            <Advanced label={s.advanced}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label className="etikett mb-1 block"><Gloss text={s.chat.topK(genTopK)} /></label>
                  <input
                    type="range"
                    min={1}
                    max={Math.max(2, stats.vocab)}
                    value={genTopK}
                    onChange={(e) => setGenTopK(Number(e.target.value))}
                    className="w-full"
                  />
                  <p className="text-[11px] text-blyant">{s.chat.topKHelp}</p>
                </div>
                <div>
                  <label className="etikett mb-1 block">{s.chat.length(genLen)}</label>
                  <input
                    type="range"
                    min={20}
                    max={240}
                    step={10}
                    value={genLen}
                    onChange={(e) => setGenLen(Number(e.target.value))}
                    className="w-full"
                  />
                  <p className="text-[11px] text-blyant">{s.chat.lengthHelp}</p>
                </div>
              </div>
            </Advanced>

            <button onClick={runGenerate} disabled={genLoading} className="knapp knapp-blekk">
              {genLoading ? s.chat.thinking : s.chat.generate}
            </button>

            {/* svaret kjem på tavla */}
            <div className="tavle p-4">
              <div className="mb-2 font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-kritt/70">
                {s.chat.answerLabel}
              </div>
              <p className="min-h-8 whitespace-pre-wrap font-mono text-sm leading-relaxed text-kritt">
                <Utskrift text={chatShown} seed={chatSeed} />
                <span className="animate-pulse text-tusj">▍</span>
              </p>
              {chatShown && (
                <p className="mt-2 text-[11px] leading-relaxed text-kritt/55">{s.seedLegend}</p>
              )}
            </div>
          </Card>
        </Section>

        {/* RLHF */}
        <Section
          id="rlhf"
          step={8}
          title={s.rlhf.sectionTitle}
          intro={s.rlhf.sectionIntro}
        >
          <Rlhf rlhf={rlhf} examples={examples} s={s} onResetTuning={onResetTuning} />
        </Section>

        {/* Ærlig note – lærerens rettepenn, etter at eleven har prøvd modellen */}
        <section className="rounded-[3px] border-2 border-rettepenn bg-white p-5 text-sm leading-relaxed">
          <b className="text-rettepenn">{s.warning.lead}</b>
          <Gloss text={s.warning.body} />
        </section>

        {/* Eigen tekst */}
        <Section
          id="eigentekst"
          step={9}
          title={s.extra.title}
          intro={s.extra.intro}
        >
          <Card>
            {/* Nedtrekkslista fyller berre tekstfeltet – å byggja på nytt er
                framleis eit eige, medvite trykk på knappen under. */}
            <div className="mb-4 max-w-md">
              <label className="etikett mb-1 block" htmlFor="eksempeltekst">
                {s.extra.sampleLabel}
              </label>
              <select
                id="eksempeltekst"
                value={sampleId}
                disabled={running}
                onChange={(e) => {
                  const id = e.target.value as EksId | "";
                  const valgt = EKSEMPELTEKSTER.find((t) => t.id === id);
                  setSampleId(id);
                  if (valgt) setExtraText(valgt.tekst);
                }}
                className="felt text-sm disabled:opacity-50"
              >
                <option value="">{s.extra.samplePlaceholder}</option>
                {EKSEMPELTEKSTER.map((t) => (
                  <option key={t.id} value={t.id}>
                    {`${t.tittel} – ${t.forfattar} (${t.aar})`}
                  </option>
                ))}
              </select>
              {sampleId && (
                <p className="mt-2 font-mono text-[11px] leading-relaxed text-blyant">
                  {(() => {
                    const t = EKSEMPELTEKSTER.find((e) => e.id === sampleId)!;
                    return `${s.extra.sampleFrom(t.tittel, t.forfattar, t.aar, t.kjelde)} ${s.extra.sampleLicense[t.lisens]} ${s.extra.sampleNote}`;
                  })()}
                </p>
              )}
            </div>
            <textarea
              value={extraText}
              onChange={(e) => {
                setExtraText(e.target.value);
                setSampleId("");
              }}
              rows={5}
              disabled={running}
              placeholder={s.extra.placeholder}
              className="felt resize-y disabled:opacity-50"
            />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <span className="font-mono text-xs text-blyant">{s.extra.charsNote(activeCorpus.length)}</span>
              <button
                onClick={rebuildWithExtraText}
                disabled={running}
                className="knapp knapp-blekk knapp-sm"
              >
                {s.extra.rebuild}
              </button>
            </div>
          </Card>
        </Section>

        {/* Les meir: baksida av kladdeboka */}
        <Section id="lesmer" step={10} title={s.readMore.title} intro={s.readMore.intro} fold={s.fold}>
          <Leseliste s={s.readMore} />
        </Section>

        {/* Ordliste: glosene aller bakarst, same kjelde som gloselappane */}
        <Section
          id="ordliste"
          step={11}
          title={s.ordliste.title}
          intro={s.ordliste.intro}
          fold={s.fold}
        >
          <Ordliste s={s.ordliste} />
        </Section>
      </main>

      <footer className="border-t-2 border-blekk py-8">
        <div className="mx-auto max-w-4xl px-4 text-center font-mono text-[11px] leading-relaxed text-blyant">
          <Gloss text={s.footer.line1} />
          <br />
          {s.footer.line2}
          <br />
          {s.footer.line3}
          {/* Dukkar opp så snart det finst noko trent, men ikkje midt i treninga:
              eksporten blokkerer tråden ein augneblink, og vektene ville flytta seg. */}
          {step > 0 && !running && (
            <>
            {/* Rekneark-modellen reknar eitt breitt lag, ikkje ein rutar med
                ekspertar. Då tilbyr vi han ikkje, i staden for å lasta ned ei
                fil som reknar på ein annan modell enn den som står på skjermen. */}
            <div className="mt-6">
              {moe ? (
                <div className="mx-auto max-w-md"><Gloss text={s.footer.excelMoe} /></div>
              ) : (
                <>
                  <button
                    onClick={onExcelClick}
                    disabled={eggState === "working"}
                    className="knapp knapp-omriss knapp-sm"
                  >
                    {s.footer.excel}
                  </button>
                  <div className="mx-auto mt-2 max-w-md">
                    {eggState === "idle" && s.footer.excelHint}
                    {eggState === "working" && s.footer.excelBusy}
                    {eggState === "done" && (
                      <span className="handnotat text-base">✓ {s.footer.excelDone}</span>
                    )}
                  </div>
                </>
              )}
            </div>
            <div className="mt-6">
              <button
                onClick={onGgufClick}
                disabled={ggufState === "working"}
                className="knapp knapp-omriss knapp-sm"
              >
                {s.footer.gguf}
              </button>
              <div className="mx-auto mt-2 max-w-md">
                {ggufState === "idle" && <Gloss text={s.footer.ggufHint} />}
                {ggufState === "working" && s.footer.ggufBusy}
                {ggufState === "done" && (
                  <span className="handnotat text-base">✓ {s.footer.ggufDone}</span>
                )}
              </div>
            </div>
            </>
          )}
        </div>
      </footer>
    </div>
    </GlossLang.Provider>
  );
}
