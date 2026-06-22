import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cn } from "@/utils/cn";
import {
  Adam,
  Transformer,
  generate,
  mulberry32,
  trainStep,
} from "@/lib/ml";
import { buildTokenizer, corpus } from "@/lib/corpus";
import LossChart from "@/components/LossChart";
import Architecture from "@/components/Architecture";
import { Section, Card } from "@/components/ui";
import Rlhf from "@/components/Rlhf";
import { useRlhf } from "@/lib/useRlhf";

const MAX_STEPS = 3500;
const CHUNK = 6;

type PresetKey = "liten" | "mellom" | "stor";

const PRESETS: Record<
  PresetKey,
  { dim: number; nLayer: number; nHead: number; seqLen: number; ffnMult: number; label: string }
> = {
  liten: { dim: 48, nLayer: 2, nHead: 2, seqLen: 32, ffnMult: 4, label: "Liten – raskast" },
  mellom: { dim: 64, nLayer: 3, nHead: 2, seqLen: 40, ffnMult: 4, label: "Mellom – balanse" },
  stor: { dim: 96, nLayer: 4, nHead: 4, seqLen: 48, ffnMult: 4, label: "Stor – tregast" },
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

export default function App() {
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
    const fullText = corpus + "\n" + customText;
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
  }, [preset, rlhf.reset]);

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
        "Det var ein gong",
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
  }, [cfg.batch]);

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
  const [chatPrompt, setChatPrompt] = useState("Det var ein gong");
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
  const displayTok = useMemo(() => buildTokenizer(corpus), []);
  const sampleSentence = "Noreg er eit land";
  const sampleTokens = useMemo(
    () => Array.from(sampleSentence).map((c) => ({ c, id: displayTok.stoi[c] })),
    [displayTok]
  );
  const vocabList = useMemo(() => displayTok.itos, [displayTok]);

  const stats = useMemo(() => {
    const eng = engineRef.current;
    return {
      params: paramCount,
      vocab: eng?.tokenizer.vocab ?? displayTok.vocab,
      chars: eng?.data.length ?? corpus.length,
      last: losses.length ? losses[losses.length - 1] : 0,
    };
  }, [paramCount, losses, displayTok]);

  const examples = ["Det var ein gong", "Noreg er", "Eg heiter", "Vatn er"];

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
              <div className="text-sm font-bold text-slate-900">Språkmodell-trener</div>
              <div className="text-[11px] text-slate-500">Lær AI på nynorsk – i nettlesaren</div>
            </div>
          </div>
          <a
            href="#trening"
            className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white transition hover:bg-slate-700"
          >
            Hopp til trening →
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
            Ekte trening frå null – ingen ferdig modell
          </div>
          <h1 className="max-w-3xl text-4xl font-extrabold leading-tight tracking-tight text-slate-900 sm:text-5xl">
            Bygg din eigen språkmodell på{" "}
            <span className="bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
              nynorsk
            </span>
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-600">
            Her trenar du ein ekte <b>transformator</b> (samme type som ChatGPT) heilt frå bunnen
            av – med ekte baklengs propagasjon og Adam-optimering. Alt skjer lokalt i maskina di.
            Følg med steg for steg, og sjå korleis tilfeldige tal blir til nynorsk tekst.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a href="#trening" className="rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-200 transition hover:bg-indigo-500">
              Start treninga
            </a>
            <a href="#forsta" className="rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">
              Forstå korleis det fungerer
            </a>
          </div>
          <div className="mt-8 grid max-w-2xl grid-cols-3 gap-3 text-center">
            {[
              { k: "teikn-nivå", v: "tokenisering" },
              { k: "100%", v: "i nettlesaren" },
              { k: "frå null", v: "ekte vektar" },
            ].map((s) => (
              <div key={s.v} className="rounded-xl border border-slate-200 bg-white/70 px-3 py-3">
                <div className="text-lg font-bold text-indigo-600">{s.k}</div>
                <div className="text-xs text-slate-500">{s.v}</div>
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
          title="Kva er ei språkmodell?"
          intro="Ei språkmodell lærer éin enkel ting: å gjetta kva teikn som kjem neste. Gjer vi det om og om igjen, kan ho skriva heile setningar."
        >
          <div className="grid gap-4 sm:grid-cols-3">
            {[
              { n: "1", t: "Gjet neste teikn", d: "Modellen les teksten så langt og gjet kva bokstav som bør koma neste.", i: "🔮" },
              { n: "2", t: "Mål feilen", d: "Vi samanliknar gjettinga med den ekte teksten og rekna ut tapet (loss).", i: "📏" },
              { n: "3", t: "Juster vektane", d: "Backpropagation flyttar alle vektane litt mot ei betre gjetting.", i: "🔧" },
            ].map((s) => (
              <Card key={s.n} className="text-center">
                <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-indigo-50 text-2xl">
                  {s.i}
                </div>
                <div className="text-xs font-bold uppercase tracking-wide text-indigo-500">Steg {s.n}</div>
                <div className="mt-1 font-semibold text-slate-900">{s.t}</div>
                <p className="mt-1 text-sm text-slate-600">{s.d}</p>
              </Card>
            ))}
          </div>
        </Section>

        {/* Data & tokenisering */}
        <Section
          id="data"
          step={1}
          title="Råtekst og tokenisering"
          intro="Først treng vi tekst. Her bruker vi norsk nynorsk. Datamaskina forstår ikkje bokstavar, så vi deler teksten opp i små einingar – token – og gir kvar av dei eit tal."
        >
          <Card className="space-y-5">
            <div>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="font-semibold text-slate-900">Utsnitt av treningsdataa (nynorsk)</h3>
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                  {stats.chars} teikn totalt
                </span>
              </div>
              <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-900 p-4 text-sm leading-relaxed text-slate-100">
{corpus.slice(0, 420)}…
              </pre>
            </div>

            <div>
              <h3 className="mb-2 font-semibold text-slate-900">Slik blir teksten til tal</h3>
              <p className="mb-3 text-sm text-slate-600">
                Vi delar opp setninga «{sampleSentence}» teikn for teikn. Kvart teikn får sin eigen ID:
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
                Heile teiknsettet ({vocabList.length} token = vokabularet)
              </h3>
              <div className="flex flex-wrap gap-1">
                {vocabList.map((c, i) => (
                  <span
                    key={i}
                    title={`teikn #${i}`}
                    className="inline-flex h-7 min-w-7 items-center justify-center rounded-md border border-slate-200 bg-slate-50 px-1.5 font-mono text-sm text-slate-600"
                  >
                    {charLabel(c)}
                  </span>
                ))}
              </div>
            </div>
          </Card>
        </Section>

        {/* Arkitektur */}
        <Section
          id="arkitektur"
          step={2}
          title="Modellarkitekturen"
          intro="Vi nyttar ein transformator – algoritmen bak moderne språkmodellar. Dataen renn oppover gjennom blokkane, og kvar blokk lærer noko nytt om samanhengen i teksten."
        >
          <Card>
            <Architecture layers={cfg.nLayer} heads={cfg.nHead} dim={cfg.dim} />
            <div className="mt-5 grid gap-3 text-sm text-slate-600 sm:grid-cols-2">
              <div className="rounded-xl bg-slate-50 p-3">
                <b className="text-slate-800">Kausal maskering:</b> når modellen gjet posisjon i, får ho
                berre sjå det som kom <i>før</i>. Slik lærer ho å skriva framover, ikkje å juksa.
              </div>
              <div className="rounded-xl bg-slate-50 p-3">
                <b className="text-slate-800">Fleire "hovud":</b> multi-head oppmerksomheit let
                modellen sjå på fleire ulike ting samtidig – t.d. både bokstav, ordlyd og tyding.
              </div>
            </div>
          </Card>
        </Section>

        {/* Trening */}
        <Section
          id="trening"
          step={3}
          title="Trening – sjå modellen læra"
          intro="No set vi i gong. For kvart steg gjet modellen, måler tapet, og flyttar vektane med Adam-optimering. Sjå om tapet går ned – då skjer læringa!"
        >
          <Card className="space-y-5">
            {/* kontrollar */}
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Modellstørrelse
                </label>
                <select
                  value={preset}
                  disabled={running || rlhf.dpoRunning}
                  onChange={(e) => setPreset(e.target.value as PresetKey)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm disabled:opacity-50"
                >
                  {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
                    <option key={k} value={k}>
                      {PRESETS[k].label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Minibatch: {cfg.batch}
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
                  Læringsrate: {lr.toFixed(4)}
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
                  ▶ Start trening
                </button>
              ) : (
                <button
                  onClick={stop}
                  className="rounded-xl bg-rose-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-rose-200 transition hover:bg-rose-500"
                >
                  ⏸ Stopp
                </button>
              )}
              <button
                onClick={reset}
                disabled={running}
                className="rounded-xl border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
              >
                ↺ Nullstill
              </button>
              <div className="ml-auto text-right text-sm">
                <div className="font-semibold text-slate-900">
                  Steg {step} / {MAX_STEPS}
                </div>
                <div className="text-xs text-slate-500">{stats.params.toLocaleString("nn")} parametrar</div>
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
              <h3 className="mb-2 font-semibold text-slate-900">Tap (loss) over tid</h3>
              <LossChart data={losses} />
              <p className="mt-2 text-xs text-slate-500">
                Lågare tap = betre. Ein perfekt modell ville hatt tap rundt 0. Jo raskare kurva
                søkjer nedover, jo fortare lærer modellen.
              </p>
            </div>

            {/* live-eksempel */}
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                <span className={cn("h-2 w-2 rounded-full", running ? "animate-pulse bg-emerald-500" : "bg-slate-300")} />
                Dette skriv modellen no
              </div>
              <p className="min-h-6 whitespace-pre-wrap font-mono text-sm text-slate-700">
                {currentSample || <span className="text-slate-400">Trykk «Start trening» for å sjå døme undervegs…</span>}
              </p>
            </div>
          </Card>
        </Section>

        {/* Chat / generering */}
        <Section
          id="chat"
          step={4}
          title="Prøv modellen"
          intro="Skriv ein starttekst, og lat modellen halda fram. Ho gjet eitt teikn om gongen. Små modellar gir ikkje perfekte svar – men sjå kor mykje betre det blir etter kvart som ho trenar!"
        >
          <Card className="space-y-5">
            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                Din starttekst (nynorsk)
              </label>
              <textarea
                value={chatPrompt}
                onChange={(e) => setChatPrompt(e.target.value)}
                rows={2}
                className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                placeholder="t.d. «Det var ein gong»"
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
                  Temperatur: {genTemp.toFixed(2)}
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
                <p className="text-[11px] text-slate-400">0 = trygg, høg = kreativ</p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Top-k: {genTopK}
                </label>
                <input
                  type="range"
                  min={1}
                  max={Math.max(2, stats.vocab)}
                  value={genTopK}
                  onChange={(e) => setGenTopK(Number(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <p className="text-[11px] text-slate-400">berre dei k beste vala</p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Lengd: {genLen} teikn
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
                <p className="text-[11px] text-slate-400">kor mange nye teikn</p>
              </div>
            </div>

            <button
              onClick={runGenerate}
              disabled={genLoading}
              className="rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-violet-200 transition hover:bg-violet-500 disabled:opacity-60"
            >
              {genLoading ? "Tenkjer…" : "✨ Generer tekst"}
            </button>

            <div className="rounded-xl border border-slate-200 bg-slate-900 p-4">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Svar frå modellen
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
          step={5}
          title="RLHF – lær modellen kva vi føretrekkjer"
          intro="Etter grunntreninga kan vi finjustere modellen med menneskeleg tilbakemelding. Du vel kva for eit av to framhald som er best, og modellen blir dytta mot valet ditt med DPO – forankra til ein frosen kopi av modellen."
        >
          <Rlhf rlhf={rlhf} examples={examples} />
        </Section>

        {/* Ærlig note */}
        <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-800">
          <b>Åtvaring – ærlig om kva dette er:</b> Dette er ein <i>svært liten</i> modell som blir
          trent i nettlesaren din på nokre få setningar. Ho kan ikkje måla seg med store modellar
          som ChatGPT, som er titusenvis av gonger større og trenar i veker på enorme mengder data.
          Men prinsippet er <b>nøyaktig det same</b>: ekte transformator, ekte backpropagation, ekte
          læring. Meir tekst og fleire steg gir betre resultat – prøv å lime inn eigen tekst i feltet under!
        </section>

        {/* Egen tekst */}
        <Section
          id="eigentekst"
          step={6}
          title="Legg til eigen tekst"
          intro="Meir og variert tekst gjer modellen betre. Lim inn nynorsk tekst her (t.d. frå ei bok eller noko du har skrive). Modellen blir bygd på nytt med den nye dataa."
        >
          <Card>
            <textarea
              value={extraText}
              onChange={(e) => setExtraText(e.target.value)}
              rows={5}
              disabled={running}
              placeholder="Lim inn nynorsk tekst her… (gjerne fleire avsnitt)"
              className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100 disabled:opacity-50"
            />
            <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
              <span>Teken med i tillegg til {corpus.length} faste teikn.</span>
              <button
                onClick={rebuildWithExtraText}
                disabled={running}
                className="rounded-lg bg-slate-900 px-3 py-1.5 font-semibold text-white transition hover:bg-slate-700 disabled:opacity-50"
              >
                Bygg modell på nytt
              </button>
            </div>
          </Card>
        </Section>
      </main>

      <footer className="border-t border-slate-200 bg-white py-8">
        <div className="mx-auto max-w-5xl px-4 text-center text-sm text-slate-500">
          Bygt med eigen skreve maskinlæringsmotor – transformator, autograd og Adam – heilt i JavaScript.
          <br />
          All kode og all læring skjer lokalt i din eigen nettlesar. 🇳🇴
        </div>
      </footer>
    </div>
  );
}
