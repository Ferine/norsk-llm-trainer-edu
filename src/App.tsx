import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/utils/cn";
import {
  Adam,
  Transformer,
  generate,
  generateDetailed,
  mulberry32,
  trainStep,
} from "@/lib/ml";
import { lossToFocus, meanConf, trailingMean } from "@/lib/chalk";
import { buildTokenizer, corpora } from "@/lib/corpus";
import { STRINGS, SEEDS, LANGS, type Lang, type Seeds, type Strings } from "@/lib/i18n";
import LossChart from "@/components/LossChart";
import Architecture from "@/components/Architecture";
import { Section, Card, Advanced } from "@/components/ui";
import Rlhf from "@/components/Rlhf";
import BpeLab from "@/components/BpeLab";
import Inspector from "@/components/Inspector";
import Skruer from "@/components/Skruer";
import Tavle from "@/components/Tavle";
import { useRlhf } from "@/lib/useRlhf";

const MAX_STEPS = 3500;
const CHUNK = 6;
// §5 sin målar les eit glidande snitt over dei siste stega, ikkje det rå
// siste tapet – sjå GAUGE_SMOOTH_WINDOW-bruken i `stats` under.
const GAUGE_SMOOTH_WINDOW = 20;

type PresetKey = "liten" | "mellom" | "stor";

const PRESETS: Record<
  PresetKey,
  { dim: number; nLayer: number; nHead: number; seqLen: number; ffnMult: number }
> = {
  liten: { dim: 48, nLayer: 2, nHead: 2, seqLen: 32, ffnMult: 4 },
  mellom: { dim: 64, nLayer: 3, nHead: 2, seqLen: 40, ffnMult: 4 },
  stor: { dim: 96, nLayer: 4, nHead: 4, seqLen: 48, ffnMult: 4 },
};

interface Engine {
  tokenizer: ReturnType<typeof buildTokenizer>;
  data: number[];
  model: Transformer;
  opt: Adam;
}

function charLabel(c: string): string {
  if (c === " ") return "␣";
  if (c === "\n") return "⏎";
  return c;
}

// Lærestripa: ekte utskrifter frå ei treningsøkt, skrivne fram teikn for teikn
// på linjert papir. Dei to første radene får lærarens raude bølgjestrek.
function LearningStrip({ rows, t }: { rows: Seeds["strip"]; t: Strings["hero"]["strip"] }) {
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
          <span
            className={`min-w-0 flex-1 truncate font-mono text-[13px] text-blekk ${
              i < 2 ? "rettelinje" : ""
            }`}
          >
            {shown[i]}
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
  // Byte av språk byggjer modellen på nytt – hald att til brukaren stadfestar
  const [pendingLang, setPendingLang] = useState<Lang | null>(null);
  const s = STRINGS[lang];
  const seed = SEEDS[lang];
  const activeCorpus = corpora[lang];
  const activeLocale = LANGS.find((l) => l.id === lang)!.locale;

  // ---- konfigurasjon ----
  const [preset, setPreset] = useState<PresetKey>("liten");
  const [batch, setBatch] = useState(4);
  const [lr, setLr] = useState(0.0008);
  const [extraText, setExtraText] = useState("");

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
  const timerRef = useRef<number | null>(null);
  const generateTimerRef = useRef<number | null>(null);
  const activeExtraTextRef = useRef("");
  // Treningsfart (steg per ms, glidande snitt) for «ca. X s igjen»
  const rateRef = useRef(0);
  const lastTickRef = useRef<{ t: number; step: number } | null>(null);
  const resetArmTimerRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [currentSample, setCurrentSample] = useState("");
  const [paramCount, setParamCount] = useState(0);
  const [resetArmed, setResetArmed] = useState(false);
  // Tel opp for kvar ny motor, så skrue-visualiseringa veit når ho må nullstille utvalet sitt.
  const [engineGen, setEngineGen] = useState(0);

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
      { vocab: tokenizer.vocab, dim: arch.dim, nLayer: arch.nLayer, nHead: arch.nHead, seqLen: arch.seqLen, ffnMult: arch.ffnMult },
      mulberry32(1337)
    );
    const opt = new Adam(model.params, lrRef.current);
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
    // berre arkitektur (preset) tvingar fram ein ny modell – ikkje lr/batch
  }, [preset, rlhf.reset, activeCorpus]);

  useEffect(() => {
    if (!runningRef.current) buildEngine();
  }, [buildEngine]);

  // Endra læringsrate utan å byggja modellen på nytt (mistar ikkje framdrift).
  useEffect(() => {
    lrRef.current = lr;
    if (engineRef.current) engineRef.current.opt.lr = lr;
  }, [lr]);

  // ---- treningsløkke ----
  const loop = useCallback(() => {
    timerRef.current = null;
    if (!runningRef.current) return;
    const eng = engineRef.current;
    if (!eng) return;
    const seqLen = eng.model.seqLen;
    const stepsThisChunk = Math.min(CHUNK, MAX_STEPS - stepRef.current);
    for (let i = 0; i < stepsThisChunk; i++) {
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

  const start = useCallback(() => {
    rlhf.reset();
    if (!engineRef.current || stepRef.current >= MAX_STEPS) buildEngine();
    lastTickRef.current = null; // ikkje la pausetid forureine farten
    setResetArmed(false);
    runningRef.current = true;
    setRunning(true);
    loop();
  }, [buildEngine, loop, rlhf.reset]);

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

  // Nullstilling kastar ei trena økt – krev to trykk når det finst noko å miste.
  const onResetClick = useCallback(() => {
    if (resetArmTimerRef.current !== null) {
      window.clearTimeout(resetArmTimerRef.current);
      resetArmTimerRef.current = null;
    }
    if (stepRef.current === 0 || resetArmed) {
      setResetArmed(false);
      reset();
      return;
    }
    setResetArmed(true);
    resetArmTimerRef.current = window.setTimeout(() => setResetArmed(false), 3000);
  }, [resetArmed, reset]);

  // Språkbyte kastar korpus + modell. Er det ei trena økt å miste, spør først.
  const requestLang = useCallback(
    (next: Lang) => {
      if (next === lang) return;
      if (stepRef.current > 0) {
        setPendingLang(next);
        return;
      }
      setLang(next);
    },
    [lang]
  );
  const confirmLang = useCallback(() => {
    if (pendingLang) setLang(pendingLang);
    setPendingLang(null);
  }, [pendingLang]);
  const cancelLang = useCallback(() => setPendingLang(null), []);

  const rebuildWithExtraText = useCallback(() => {
    stop();
    buildEngine(extraText);
  }, [stop, buildEngine, extraText]);

  useEffect(
    () => () => {
      runningRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      if (generateTimerRef.current !== null) window.clearTimeout(generateTimerRef.current);
      if (resetArmTimerRef.current !== null) window.clearTimeout(resetArmTimerRef.current);
    },
    []
  );

  // ---- generering / "chat" ----
  const [chatPrompt, setChatPrompt] = useState(seed.chatPrompt);
  const [chatFull, setChatFull] = useState("");
  const [chatShown, setChatShown] = useState("");
  const [chatConf, setChatConf] = useState<Float32Array>(() => new Float32Array(0));
  const [chatPromptLen, setChatPromptLen] = useState(0);
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
      const out = generateDetailed(
        eng.model,
        eng.tokenizer.decode,
        eng.tokenizer.encode,
        chatPrompt,
        { temperature: genTemp, topK: genTopK, length: genLen },
        sampleRngRef.current
      );
      setChatFull(out.text);
      setChatConf(out.conf);
      setChatPromptLen(out.promptLen);
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
      // Glidande snitt for §5 sin målar (sjå GAUGE_SMOOTH_WINDOW): éin
      // minibatch-loss hoppar med støy kvar CHUNK-oppdatering (~6 Hz), og
      // det er verken ei leseleg tavle eller eit ærleg "korleis går det no"-
      // tal. `last` over blir ikkje endra – han er framleis det rå,
      // augeblinkelege talet aksen i header viser.
      smoothed: trailingMean(losses, GAUGE_SMOOTH_WINDOW),
    };
  }, [paramCount, losses, displayTok, activeCorpus]);

  const examples = seed.examples;

  const getParams = useCallback(() => engineRef.current?.model.params ?? null, []);

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
    <div className="min-h-screen">
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
                  onClick={() => requestLang(l.id)}
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
          {/* Levande status: full breidd på eiga linje på mobil, inline til høgre elles */}
          {(running || step > 0) && (
            <div className="order-3 flex w-full items-center gap-2 border-t border-blekk/15 pt-2 font-mono text-[11px] sm:order-none sm:ml-auto sm:w-auto sm:border-0 sm:pt-0">
              <span
                className={cn(
                  "h-2 w-2 flex-none rounded-full",
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
          <a
            href="#trening"
            className="knapp knapp-blekk knapp-sm order-2 flex-none sm:order-none"
          >
            {s.header.jump}
          </a>
        </div>
        {/* Stadfesting: språkbyte kastar den trena økta – rettepennens raude strek */}
        {pendingLang && (
          <div className="border-t-2 border-rettepenn bg-white">
            <div className="mx-auto flex max-w-4xl flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2.5 sm:px-4">
              <span className="min-w-0 flex-1 text-sm leading-snug text-blekk">
                {s.header.langConfirm(
                  LANGS.find((l) => l.id === pendingLang)!.label.toLowerCase()
                )}
              </span>
              <div className="flex flex-none gap-2">
                <button onClick={confirmLang} className="knapp knapp-rettepenn knapp-sm">
                  {s.header.langConfirmYes}
                </button>
                <button onClick={cancelLang} className="knapp knapp-omriss knapp-sm">
                  {s.header.langConfirmNo}
                </button>
              </div>
            </div>
          </div>
        )}
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
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-blyant">{s.hero.para}</p>
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
          <LearningStrip rows={seed.strip} t={s.hero.strip} />
          <p className="mt-3 max-w-3xl font-mono text-[11px] leading-relaxed text-blyant">
            {s.hero.strip.caption}
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
                <p className="mt-1 text-sm leading-relaxed text-blyant">{c.d}</p>
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
              <p className="max-w-2xl text-sm leading-relaxed text-blyant">{s.data.originPara}</p>
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
            <Architecture layers={cfg.nLayer} heads={cfg.nHead} dim={cfg.dim} s={s} />
            <div className="mt-5 grid grid-cols-1 gap-3 text-sm leading-relaxed text-blyant sm:grid-cols-2">
              <div className="rounded-[2px] border border-blekk/25 bg-papir/70 p-3">
                <b className="text-blekk">{s.arch.causalTitle}</b> {s.arch.causalBody}
              </div>
              <div className="rounded-[2px] border border-blekk/25 bg-papir/70 p-3">
                <b className="text-blekk">{s.arch.headsTitle}</b> {s.arch.headsBody}
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
                onChange={(e) => setPreset(e.target.value as PresetKey)}
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
                  <label className="etikett mb-1 block">{s.train.minibatch(cfg.batch)}</label>
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
                  <label className="etikett mb-1 block">{s.train.learningRate(lr.toFixed(4))}</label>
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
                className={cn("knapp", resetArmed ? "knapp-rettepenn" : "knapp-omriss")}
              >
                {resetArmed ? s.train.resetConfirm : s.train.reset}
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
                <div className="font-mono text-xs text-blyant">{stats.params.toLocaleString(activeLocale)} {s.train.params}</div>
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

            {/* tap-graf */}
            <div>
              <h3 className="mb-2 font-semibold text-blekk">{s.train.lossHeading}</h3>
              <LossChart data={losses} loss={s.loss} />
              <p className="mt-2 text-xs leading-relaxed text-blyant">
                {s.train.lossHelp}
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

            {/* live-eksempel: eleven skriv på tavla */}
            <Tavle
              label={s.train.liveLabel}
              text={currentSample}
              placeholder={s.train.livePlaceholder}
              legend={s.train.focusLegend}
              // Målaren (både samandraget her og `gauge` under) les det
              // glatta tapet, ikkje `stats.last` – sjå kommentaren ved
              // `smoothed` i `stats`. Header-aksen over held fram med
              // `stats.last` urørt: det er den rå, augeblinkelege statusen,
              // medan denne målaren skal vere til å lese medan han oppdaterer.
              summary={s.train.focusSummary(
                Math.round(lossToFocus(stats.smoothed, stats.vocab) * 100)
              )}
              // oppgåve 5 sin opphavlege tekststil: lågare min-høgd, ingen linjeavstand
              textClassName="min-h-6 whitespace-pre-wrap font-mono text-sm text-kritt"
              // Tier 2 er halden att her: denne målaren teiknar inni sjølve
              // opplæringsløkka, appens varmaste sti (~16ms per steg). Det er
              // IKKJE eit nytt WebGL-kontekst som er dyrt – getContext("webgl2")
              // på same lerret gjev same konteksten att, kvar gong. Det som
              // faktisk vart bygd på nytt kvart steg (før det vart retta) var
              // sjølve GL-programmet og ressursane hans. Sjølv med det retta,
              // er kostnaden ved den løpande teikninga – texElementImage2D +
              // shader kvart steg – aldri målt her, og tier 2 er uansett
              // uverifisert (sjå tier-deteksjonen i chalk.ts). Difor står
              // lerretet halde att her til nokon har målt steg/sekund med og
              // utan. Oppgåve 7 sin målar (under) skil seg frå denne: han
              // teiknar éin gong per generering, ikkje per treningssteg, så
              // han held fram med tier 2 (når han er tvinga på med ?tier=2 og
              // nettlesaren støttar det).
              noCanvas
              // måleren gjeld berre når det finst eit ekte tap å måle mot
              gauge={
                losses.length > 0
                  ? { kind: "loss", value: stats.smoothed, vocab: stats.vocab }
                  : undefined
              }
            >
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  running ? "animate-pulse bg-tusj" : "bg-kritt/30"
                )}
              />
            </Tavle>
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

        {/* Ærlig note – lærerens rettepenn, plassert FØR eleven prøver seg */}
        <section className="rounded-[3px] border-2 border-rettepenn bg-white p-5 text-sm leading-relaxed">
          <b className="text-rettepenn">{s.warning.lead}</b>
          {s.warning.body}
        </section>

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
              <label className="etikett mb-1 block">{s.chat.temp(genTemp.toFixed(2))}</label>
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
                  <label className="etikett mb-1 block">{s.chat.topK(genTopK)}</label>
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
            <Tavle
              label={s.chat.answerLabel}
              text={chatShown}
              placeholder=""
              legend={s.chat.confLegend}
              summary={s.chat.confSummary(Math.round(meanConf(chatConf) * 100))}
              gauge={
                chatConf.length > 0
                  ? { kind: "conf", conf: chatConf, promptLen: chatPromptLen }
                  : undefined
              }
            />
          </Card>
        </Section>

        {/* RLHF */}
        <Section
          id="rlhf"
          step={8}
          title={s.rlhf.sectionTitle}
          intro={s.rlhf.sectionIntro}
        >
          <Rlhf rlhf={rlhf} examples={examples} s={s} />
        </Section>

        {/* Eigen tekst */}
        <Section
          id="eigentekst"
          step={9}
          title={s.extra.title}
          intro={s.extra.intro}
        >
          <Card>
            <textarea
              value={extraText}
              onChange={(e) => setExtraText(e.target.value)}
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
      </main>

      <footer className="border-t-2 border-blekk py-8">
        <div className="mx-auto max-w-4xl px-4 text-center font-mono text-[11px] leading-relaxed text-blyant">
          {s.footer.line1}
          <br />
          {s.footer.line2}
        </div>
      </footer>
    </div>
  );
}
