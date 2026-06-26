import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/utils/cn";
import {
  Adam,
  Transformer,
  generate,
  mulberry32,
  trainStep,
} from "@/lib/ml";
import { buildTokenizer, corpora } from "@/lib/corpus";
import { STRINGS, SEEDS, LANGS, type Lang } from "@/lib/i18n";
import LossChart from "@/components/LossChart";
import Architecture from "@/components/Architecture";
import { Section, Card } from "@/components/ui";
import Rlhf from "@/components/Rlhf";
import BpeLab from "@/components/BpeLab";
import Inspector from "@/components/Inspector";
import { useRlhf } from "@/lib/useRlhf";

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

  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [currentSample, setCurrentSample] = useState("");
  const [paramCount, setParamCount] = useState(0);

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
    runningRef.current = true;
    setRunning(true);
    loop();
  }, [buildEngine, loop, rlhf.reset]);

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

  const rebuildWithExtraText = useCallback(() => {
    stop();
    buildEngine(extraText);
  }, [stop, buildEngine, extraText]);

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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-indigo-50/40 text-slate-800">
      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-slate-200/70 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-lg shadow">
              🧠
            </div>
            <div className="leading-tight">
              <div className="text-sm font-bold text-slate-900">{s.header.title}</div>
              <div className="text-[11px] text-slate-500">{s.header.subtitle}</div>
            </div>
            <div className="ml-3 inline-flex overflow-hidden rounded-lg border border-slate-300 text-xs font-semibold">
              {LANGS.map((l) => (
                <button
                  key={l.id}
                  onClick={() => setLang(l.id)}
                  disabled={running || rlhf.dpoRunning}
                  className={cn(
                    "px-2.5 py-1 transition disabled:opacity-50",
                    lang === l.id ? "bg-slate-900 text-white" : "bg-white text-slate-600 hover:bg-slate-50"
                  )}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>
          <a
            href="#trening"
            className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700"
          >
            {s.header.jump}
          </a>
        </div>
      </header>

      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 opacity-60">
          <div className="absolute -left-20 top-0 h-72 w-72 rounded-full bg-indigo-300/40 blur-3xl" />
          <div className="absolute right-0 top-10 h-72 w-72 rounded-full bg-violet-300/40 blur-3xl" />
        </div>
        <div className="relative mx-auto max-w-5xl px-4 py-14 sm:py-20">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
            <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
            {s.hero.badge}
          </div>
          <h1 className="max-w-3xl text-4xl font-extrabold leading-tight tracking-tight text-slate-900 sm:text-5xl">
            {s.hero.h1Pre}{" "}
            <span className="bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
              {s.hero.h1Lang}
            </span>
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-600">
            {s.hero.para}
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a href="#trening" className="rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-200 transition hover:bg-indigo-500">
              {s.hero.ctaStart}
            </a>
            <a href="#forsta" className="rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">
              {s.hero.ctaUnderstand}
            </a>
          </div>
          <div className="mt-8 grid max-w-2xl grid-cols-3 gap-3 text-center">
            {s.hero.stats.map((st) => (
              <div key={st.v} className="rounded-xl border border-slate-200 bg-white/70 px-3 py-3">
                <div className="text-lg font-bold text-indigo-600">{st.k}</div>
                <div className="text-xs text-slate-500">{st.v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-5xl space-y-20 px-4 py-12">
        {/* Forstå */}
        <Section
          id="forsta"
          step={0}
          title={s.understand.title}
          intro={s.understand.intro}
        >
          <div className="grid gap-4 sm:grid-cols-3">
            {s.understand.cards.map((c, i) => (
              <Card key={i} className="text-center">
                <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-50 text-2xl">
                  {c.i}
                </div>
                <div className="text-xs font-bold uppercase tracking-wide text-indigo-500">Steg {i + 1}</div>
                <div className="mt-1 font-semibold text-slate-900">{c.t}</div>
                <p className="mt-1 text-sm text-slate-600">{c.d}</p>
              </Card>
            ))}
          </div>
        </Section>

        {/* Data & tokenisering */}
        <Section
          id="data"
          step={1}
          title={s.data.title}
          intro={s.data.intro}
        >
          <Card className="space-y-5">
            <div>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="font-semibold text-slate-900">{s.data.snippetHeading}</h3>
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                  {s.data.charsTotal(stats.chars)}
                </span>
              </div>
              <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-900 p-4 text-sm leading-relaxed text-slate-100">
{activeCorpus.slice(0, 420)}…
              </pre>
            </div>

            <div>
              <h3 className="mb-2 font-semibold text-slate-900">{s.data.howHeading}</h3>
              <p className="mb-3 text-sm text-slate-600">
                {s.data.howPara(sampleSentence)}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {sampleTokens.map((t, i) => (
                  <span
                    key={i}
                    className="inline-flex flex-col items-center rounded-lg border border-indigo-200 bg-indigo-50 px-2 py-1 text-center"
                  >
                    <span className="font-mono text-base font-semibold text-indigo-700">
                      {charLabel(t.c)}
                    </span>
                    <span className="text-[10px] text-slate-500">#{t.id}</span>
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-2 font-semibold text-slate-900">
                {s.data.vocabHeading(vocabList.length)}
              </h3>
              <div className="flex flex-wrap gap-1">
                {vocabList.map((c, i) => (
                  <span
                    key={i}
                    title={s.data.charTooltip(i)}
                    className="inline-flex h-7 min-w-7 items-center justify-center rounded-md border border-slate-200 bg-slate-50 px-1.5 font-mono text-sm text-slate-600"
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
          step={2}
          title={s.bpe.title}
          intro={s.bpe.intro}
        >
          <Card>
            <BpeLab corpus={activeCorpus} sampleSentence={sampleSentence} s={s.bpe} />
          </Card>
        </Section>

        {/* Arkitektur */}
        <Section
          id="arkitektur"
          step={3}
          title={s.arch.title}
          intro={s.arch.intro}
        >
          <Card>
            <Architecture layers={cfg.nLayer} heads={cfg.nHead} dim={cfg.dim} s={s} />
            <div className="mt-5 grid gap-3 text-sm text-slate-600 sm:grid-cols-2">
              <div className="rounded-xl bg-slate-50 p-3">
                <b className="text-slate-800">{s.arch.causalTitle}</b> {s.arch.causalBody}
              </div>
              <div className="rounded-xl bg-slate-50 p-3">
                <b className="text-slate-800">{s.arch.headsTitle}</b> {s.arch.headsBody}
              </div>
            </div>
          </Card>
        </Section>

        {/* Trening */}
        <Section
          id="trening"
          step={4}
          title={s.train.title}
          intro={s.train.intro}
        >
          <Card className="space-y-5">
            {/* kontrollar */}
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {s.train.modelSize}
                </label>
                <select
                  value={preset}
                  disabled={running || rlhf.dpoRunning}
                  onChange={(e) => setPreset(e.target.value as PresetKey)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm disabled:opacity-50"
                >
                  {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
                    <option key={k} value={k}>
                      {s.train.presets[k]}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {s.train.minibatch(cfg.batch)}
                </label>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={batch}
                  disabled={running || rlhf.dpoRunning}
                  onChange={(e) => setBatch(Number(e.target.value))}
                  className="w-full accent-indigo-600 disabled:opacity-50"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {s.train.learningRate(lr.toFixed(4))}
                </label>
                <input
                  type="range"
                  min={0.0001}
                  max={0.003}
                  step={0.0001}
                  value={lr}
                  disabled={running || rlhf.dpoRunning}
                  onChange={(e) => setLr(Number(e.target.value))}
                  className="w-full accent-indigo-600 disabled:opacity-50"
                />
              </div>
            </div>

            {/* knappar */}
            <div className="flex flex-wrap items-center gap-3">
              {!running ? (
                <button
                  onClick={start}
                  disabled={rlhf.dpoRunning}
                  className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-indigo-200 transition hover:bg-indigo-500 disabled:opacity-50"
                >
                  {s.train.start}
                </button>
              ) : (
                <button
                  onClick={stop}
                  className="rounded-xl bg-rose-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-rose-200 transition hover:bg-rose-500"
                >
                  {s.train.stop}
                </button>
              )}
              <button
                onClick={reset}
                disabled={running}
                className="rounded-xl border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
              >
                {s.train.reset}
              </button>
              <div className="ml-auto text-right text-sm">
                <div className="font-semibold text-slate-900">
                  {s.train.step(step, MAX_STEPS)}
                </div>
                <div className="text-xs text-slate-500">{stats.params.toLocaleString(activeLocale)} {s.train.params}</div>
              </div>
            </div>

            {/* framdrift */}
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all"
                style={{ width: `${(step / MAX_STEPS) * 100}%` }}
              />
            </div>

            {/* tap-graf */}
            <div>
              <h3 className="mb-2 font-semibold text-slate-900">{s.train.lossHeading}</h3>
              <LossChart data={losses} loss={s.loss} />
              <p className="mt-2 text-xs text-slate-500">
                {s.train.lossHelp}
              </p>
            </div>

            {/* live-eksempel */}
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                <span className={cn("h-2 w-2 rounded-full", running ? "animate-pulse bg-emerald-500" : "bg-slate-300")} />
                {s.train.liveLabel}
              </div>
              <p className="min-h-6 whitespace-pre-wrap font-mono text-sm text-slate-700">
                {currentSample || <span className="text-slate-400">{s.train.livePlaceholder}</span>}
              </p>
            </div>
          </Card>
        </Section>

        {/* Se inni modellen */}
        <Section
          id="inspect"
          step={5}
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
          step={6}
          title={s.chat.title}
          intro={s.chat.intro}
        >
          <Card className="space-y-5">
            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                {s.chat.promptLabel}
              </label>
              <textarea
                value={chatPrompt}
                onChange={(e) => setChatPrompt(e.target.value)}
                rows={2}
                className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                placeholder={s.chat.promptPlaceholder}
              />
              <div className="mt-2 flex flex-wrap gap-2">
                {examples.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => setChatPrompt(ex)}
                    className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 transition hover:border-indigo-300 hover:text-indigo-600"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {s.chat.temp(genTemp.toFixed(2))}
                </label>
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={genTemp}
                  onChange={(e) => setGenTemp(Number(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <p className="text-[11px] text-slate-400">{s.chat.tempHelp}</p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {s.chat.topK(genTopK)}
                </label>
                <input
                  type="range"
                  min={1}
                  max={Math.max(2, stats.vocab)}
                  value={genTopK}
                  onChange={(e) => setGenTopK(Number(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <p className="text-[11px] text-slate-400">{s.chat.topKHelp}</p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {s.chat.length(genLen)}
                </label>
                <input
                  type="range"
                  min={20}
                  max={240}
                  step={10}
                  value={genLen}
                  onChange={(e) => setGenLen(Number(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <p className="text-[11px] text-slate-400">{s.chat.lengthHelp}</p>
              </div>
            </div>

            <button
              onClick={runGenerate}
              disabled={genLoading}
              className="rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-violet-200 transition hover:bg-violet-500 disabled:opacity-60"
            >
              {genLoading ? s.chat.thinking : s.chat.generate}
            </button>

            <div className="rounded-xl border border-slate-200 bg-slate-900 p-4">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                {s.chat.answerLabel}
              </div>
              <p className="min-h-8 whitespace-pre-wrap font-mono text-sm leading-relaxed text-emerald-100">
                {chatShown}
                <span className="animate-pulse text-emerald-300">▍</span>
              </p>
            </div>
          </Card>
        </Section>

        {/* RLHF */}
        <Section
          id="rlhf"
          step={7}
          title={s.rlhf.sectionTitle}
          intro={s.rlhf.sectionIntro}
        >
          <Rlhf rlhf={rlhf} examples={examples} s={s} />
        </Section>

        {/* Ærlig note */}
        <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
          <b>{s.warning.lead}</b>{s.warning.body}
        </section>

        {/* Eigen tekst */}
        <Section
          id="eigentekst"
          step={8}
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
              className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100 disabled:opacity-50"
            />
            <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
              <span>{s.extra.charsNote(activeCorpus.length)}</span>
              <button
                onClick={rebuildWithExtraText}
                disabled={running}
                className="rounded-lg bg-slate-900 px-3 py-1.5 font-semibold text-white transition hover:bg-slate-700 disabled:opacity-50"
              >
                {s.extra.rebuild}
              </button>
            </div>
          </Card>
        </Section>
      </main>

      <footer className="border-t border-slate-200 bg-white py-8">
        <div className="mx-auto max-w-5xl px-4 text-center text-sm text-slate-500">
          {s.footer.line1}
          <br />
          {s.footer.line2}
        </div>
      </footer>
    </div>
  );
}
