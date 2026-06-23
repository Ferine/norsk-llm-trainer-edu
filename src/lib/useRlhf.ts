import { useCallback, useEffect, useRef, useState } from "react";
import {
  Adam,
  Transformer,
  cloneTransformer,
  dpoStep,
  mulberry32,
  sampleTokens,
  type PrefPair,
} from "@/lib/ml";
import type { Tokenizer } from "@/lib/corpus";

const BETA = 0.1;
const DPO_LR = 1e-4;
const PAIR_LEN = 48;
const LIVE_BURST = 5;
const LIVE_MINIBATCH = 4;
const TRAIN_MORE_STEPS = 60;
const TRAIN_CHUNK = 6;
const TRAIN_MINIBATCH = 4;
const PAIR_RETRY = 4;

export interface RlhfPair {
  text: string;
  promptIds: number[];
  contIds: number[];
}
export interface RlhfMetrics {
  loss: number;
  margin: number;
  winRate: number;
  count: number;
}

interface Args {
  getModel: () => Transformer | null;
  getTokenizer: () => Tokenizer | null;
  isTrained: () => boolean;
  baseRunning: boolean;
}

export function useRlhf({ getModel, getTokenizer, isTrained, baseRunning }: Args) {
  const referenceRef = useRef<Transformer | null>(null);
  const dpoOptRef = useRef<Adam | null>(null);
  const bufferRef = useRef<PrefPair[]>([]);
  const lossesRef = useRef<number[]>([]);
  const pairRngRef = useRef<() => number>(mulberry32(2027));
  const timerRef = useRef<number | null>(null);
  const genTimerRef = useRef<number | null>(null);
  const dpoStepCountRef = useRef(0);
  const trainStartRef = useRef(0);
  const runningRef = useRef(false);

  const [started, setStarted] = useState(false);
  const [dpoRunning, setDpoRunning] = useState(false);
  const [untrainedHint, setUntrainedHint] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [temp, setTemp] = useState(1.0);
  const [pairA, setPairA] = useState<RlhfPair | null>(null);
  const [pairB, setPairB] = useState<RlhfPair | null>(null);
  const [generating, setGenerating] = useState(false);
  const [losses, setLosses] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<RlhfMetrics>({ loss: 0, margin: 0, winRate: 0, count: 0 });

  const samplePair = useCallback((): { a: RlhfPair; b: RlhfPair } | null => {
    const model = getModel();
    const tok = getTokenizer();
    if (!model || !tok) return null;
    const opts = { temperature: Math.max(0.1, temp), topK: model.vocab, length: PAIR_LEN };
    const mk = (): RlhfPair => {
      const { promptIds, contIds } = sampleTokens(model, tok.encode, prompt, opts, pairRngRef.current);
      return { text: tok.decode(contIds), promptIds, contIds };
    };
    const a = mk();
    let b = mk();
    for (let r = 0; r < PAIR_RETRY && b.text === a.text; r++) b = mk();
    return { a, b };
  }, [getModel, getTokenizer, prompt, temp]);

  const generatePair = useCallback(() => {
    if (genTimerRef.current !== null) window.clearTimeout(genTimerRef.current);
    setGenerating(true);
    // yield to the browser so the "lagar par…" state can paint before heavy sampling
    genTimerRef.current = window.setTimeout(() => {
      genTimerRef.current = null;
      const pair = samplePair();
      if (pair) {
        setPairA(pair.a);
        setPairB(pair.b);
      }
      setGenerating(false);
    }, 10);
  }, [samplePair]);

  const start = useCallback(() => {
    const model = getModel();
    if (!model) return;
    referenceRef.current = cloneTransformer(model);
    dpoOptRef.current = new Adam(model.params, DPO_LR);
    bufferRef.current = [];
    lossesRef.current = [];
    dpoStepCountRef.current = 0;
    pairRngRef.current = mulberry32(2027);
    setUntrainedHint(!isTrained());
    setLosses([]);
    setMetrics({ loss: 0, margin: 0, winRate: 0, count: 0 });
    setStarted(true);
    generatePair();
  }, [getModel, isTrained, generatePair]);

  const runBurst = useCallback(
    (steps: number, minibatch: number) => {
      const model = getModel();
      const ref = referenceRef.current;
      const opt = dpoOptRef.current;
      if (!model || !ref || !opt || bufferRef.current.length === 0) return;
      let last = { loss: 0, margin: 0, winRate: 0 };
      for (let i = 0; i < steps; i++) {
        last = dpoStep(model, ref, opt, bufferRef.current, minibatch, BETA, pairRngRef.current);
        lossesRef.current.push(last.loss);
        dpoStepCountRef.current++;
      }
      setLosses(lossesRef.current.slice());
      setMetrics({ loss: last.loss, margin: last.margin, winRate: last.winRate, count: bufferRef.current.length });
    },
    [getModel]
  );

  const choose = useCallback(
    (winner: "A" | "B") => {
      const a = pairA;
      const b = pairB;
      if (!a || !b) return;
      const chosen = winner === "A" ? a : b;
      const rejected = winner === "A" ? b : a;
      bufferRef.current.push({
        promptIds: chosen.promptIds,
        chosenIds: chosen.contIds,
        rejectedIds: rejected.contIds,
      });
      runBurst(LIVE_BURST, LIVE_MINIBATCH);
      generatePair();
    },
    [pairA, pairB, runBurst, generatePair]
  );

  const skip = useCallback(() => generatePair(), [generatePair]);

  const trainLoop = useCallback(() => {
    timerRef.current = null;
    if (!runningRef.current) return;
    const model = getModel();
    const ref = referenceRef.current;
    const opt = dpoOptRef.current;
    if (!model || !ref || !opt || bufferRef.current.length === 0) {
      runningRef.current = false;
      setDpoRunning(false);
      return;
    }
    const done = dpoStepCountRef.current - trainStartRef.current;
    const chunk = Math.min(TRAIN_CHUNK, TRAIN_MORE_STEPS - done);
    let last = { loss: 0, margin: 0, winRate: 0 };
    for (let i = 0; i < chunk; i++) {
      last = dpoStep(model, ref, opt, bufferRef.current, TRAIN_MINIBATCH, BETA, pairRngRef.current);
      lossesRef.current.push(last.loss);
      dpoStepCountRef.current++;
    }
    setLosses(lossesRef.current.slice());
    setMetrics({ loss: last.loss, margin: last.margin, winRate: last.winRate, count: bufferRef.current.length });
    if (dpoStepCountRef.current - trainStartRef.current >= TRAIN_MORE_STEPS) {
      runningRef.current = false;
      setDpoRunning(false);
      return;
    }
    timerRef.current = window.setTimeout(trainLoop, 0);
  }, [getModel]);

  const trainMore = useCallback(() => {
    if (bufferRef.current.length === 0 || runningRef.current) return;
    trainStartRef.current = dpoStepCountRef.current;
    runningRef.current = true;
    setDpoRunning(true);
    trainLoop();
  }, [trainLoop]);

  const stopTrainMore = useCallback(() => {
    runningRef.current = false;
    setDpoRunning(false);
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const resetTuning = useCallback(() => {
    stopTrainMore();
    const model = getModel();
    const ref = referenceRef.current;
    if (model && ref) for (let i = 0; i < model.params.length; i++) model.params[i].d.set(ref.params[i].d);
    if (model) dpoOptRef.current = new Adam(model.params, DPO_LR);
    bufferRef.current = [];
    lossesRef.current = [];
    dpoStepCountRef.current = 0;
    setLosses([]);
    setMetrics({ loss: 0, margin: 0, winRate: 0, count: 0 });
    generatePair();
  }, [getModel, generatePair, stopTrainMore]);

  const reset = useCallback(() => {
    runningRef.current = false;
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (genTimerRef.current !== null) {
      window.clearTimeout(genTimerRef.current);
      genTimerRef.current = null;
    }
    referenceRef.current = null;
    dpoOptRef.current = null;
    bufferRef.current = [];
    lossesRef.current = [];
    dpoStepCountRef.current = 0;
    setStarted(false);
    setDpoRunning(false);
    setUntrainedHint(false);
    setPairA(null);
    setPairB(null);
    setGenerating(false);
    setLosses([]);
    setMetrics({ loss: 0, margin: 0, winRate: 0, count: 0 });
  }, []);

  useEffect(
    () => () => {
      runningRef.current = false;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      if (genTimerRef.current !== null) window.clearTimeout(genTimerRef.current);
    },
    []
  );

  return {
    started,
    dpoRunning,
    baseRunning,
    untrainedHint,
    prompt,
    setPrompt,
    temp,
    setTemp,
    pairA,
    pairB,
    generating,
    losses,
    metrics,
    start,
    generatePair,
    choose,
    skip,
    trainMore,
    stopTrainMore,
    resetTuning,
    reset,
  };
}
