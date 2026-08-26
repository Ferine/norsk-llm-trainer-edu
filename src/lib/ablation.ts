import { buildTokenizer } from "./corpus.js";
import {
  Adam,
  NGRAM_DEFAULT,
  Transformer,
  evalLoss,
  mulberry32,
  trainStepDetailed,
  type ModelConfig,
  type NgramConfig,
} from "./ml.js";

export type AblationVariant = "baseline" | "trigram";
export type AblationBaseConfig = Omit<ModelConfig, "vocab" | "ngram">;

export interface NgramAblationOptions {
  trainText: string;
  holdoutText: string;
  baseConfig: AblationBaseConfig;
  steps: number;
  batchSize: number;
  learningRate: number;
  seeds: number[];
  evalBatches?: number;
  ngram?: NgramConfig;
}

export interface AblationRun {
  variant: AblationVariant;
  seed: number;
  params: number;
  ngramParams: number;
  tailTrainLoss: number;
  heldoutLoss: number;
  msPerStep: number;
  gradP99: number;
  clipRate: number;
  maxActivation: number;
}

export interface AblationSummary {
  variant: AblationVariant;
  runs: number;
  params: number;
  ngramParams: number;
  tailTrainLoss: number;
  heldoutLoss: number;
  msPerStep: number;
  gradP99: number;
  clipRate: number;
  maxActivation: number;
}

export interface NgramAblationResult {
  tokenizer: {
    vocab: number;
    trainCharacters: number;
    trainTokens: number;
    holdoutCharacters: number;
    holdoutTokens: number;
    oneCharacterPerToken: true;
  };
  config: {
    base: AblationBaseConfig;
    ngram: NgramConfig;
    steps: number;
    batchSize: number;
    learningRate: number;
    seeds: number[];
    evalBatches: number;
  };
  sharedInitMaxDiff: number;
  runs: AblationRun[];
  baseline: AblationSummary;
  trigram: AblationSummary;
  delta: {
    params: number;
    paramsPct: number;
    heldoutLoss: number;
    heldoutLossPct: number;
    msPerStepPct: number;
    gradP99Pct: number;
    clipRate: number;
    maxActivationPct: number;
  };
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function percentile(xs: number[], q: number): number {
  const sorted = xs.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))];
}

function pct(next: number, base: number): number {
  return base === 0 ? (next === 0 ? 0 : Number.POSITIVE_INFINITY) : ((next - base) / base) * 100;
}

function now(): number {
  return globalThis.performance?.now?.() ?? Date.now();
}

function validateOptions(o: NgramAblationOptions): void {
  if (Array.from(o.trainText).length < 2) throw new RangeError("trainText needs at least two characters");
  if (Array.from(o.holdoutText).length < 2)
    throw new RangeError("holdoutText needs at least two characters");
  if (!Number.isInteger(o.steps) || o.steps < 1) throw new RangeError("steps must be positive");
  if (!Number.isInteger(o.batchSize) || o.batchSize < 1)
    throw new RangeError("batchSize must be positive");
  if (!Number.isFinite(o.learningRate) || o.learningRate <= 0)
    throw new RangeError("learningRate must be positive");
  if (o.seeds.length < 1 || o.seeds.some((s) => !Number.isInteger(s)))
    throw new RangeError("seeds must contain integers");
}

function sharedInitDiff(a: Transformer, b: Transformer): number {
  // Trigramtabellen blir med vilje lagd sist i params; alt før henne er den
  // eksakte baseline-modellen og skal vera identisk ved same frø.
  if (b.params.length !== a.params.length + 1)
    throw new Error("the trigram variant should add exactly one parameter tensor");
  let max = 0;
  for (let p = 0; p < a.params.length; p++) {
    if (a.params[p].d.length !== b.params[p].d.length)
      throw new Error(`shared parameter ${p} changed shape`);
    for (let i = 0; i < a.params[p].d.length; i++)
      max = Math.max(max, Math.abs(a.params[p].d[i] - b.params[p].d[i]));
  }
  return max;
}

function runOne(
  variant: AblationVariant,
  seed: number,
  model: Transformer,
  trainData: number[],
  holdoutData: number[],
  o: NgramAblationOptions,
  evalBatches: number
): AblationRun {
  const opt = new Adam(model.params, o.learningRate);
  // Same sampled training windows for both sides of every pair.
  const trainRng = mulberry32((seed ^ 0x9e3779b9) >>> 0);
  const losses: number[] = [];
  const grads: number[] = [];
  let clipped = 0;
  let maxActivation = 0;
  const started = now();
  for (let step = 0; step < o.steps; step++) {
    const m = trainStepDetailed(
      model,
      opt,
      trainData,
      o.baseConfig.seqLen,
      o.batchSize,
      trainRng
    );
    losses.push(m.loss);
    grads.push(m.gradNorm);
    if (m.clipped) clipped++;
    maxActivation = Math.max(maxActivation, m.maxActivation);
  }
  const elapsed = now() - started;
  const tail = losses.slice(-Math.max(1, Math.ceil(losses.length * 0.2)));
  // Same held-out windows for both variants, independent of their training RNG.
  const heldoutLoss = evalLoss(
    model,
    holdoutData,
    o.baseConfig.seqLen,
    evalBatches,
    mulberry32((seed ^ 0x85ebca6b) >>> 0)
  );
  return {
    variant,
    seed,
    params: model.paramCount(),
    ngramParams: model.ngramParamCount(),
    tailTrainLoss: mean(tail),
    heldoutLoss,
    msPerStep: elapsed / o.steps,
    gradP99: percentile(grads, 0.99),
    clipRate: clipped / o.steps,
    maxActivation,
  };
}

function summarize(variant: AblationVariant, runs: AblationRun[]): AblationSummary {
  const rs = runs.filter((r) => r.variant === variant);
  if (rs.length < 1) throw new Error(`no ${variant} runs to summarize`);
  return {
    variant,
    runs: rs.length,
    params: rs[0].params,
    ngramParams: rs[0].ngramParams,
    tailTrainLoss: mean(rs.map((r) => r.tailTrainLoss)),
    heldoutLoss: mean(rs.map((r) => r.heldoutLoss)),
    msPerStep: mean(rs.map((r) => r.msPerStep)),
    gradP99: mean(rs.map((r) => r.gradP99)),
    clipRate: mean(rs.map((r) => r.clipRate)),
    maxActivation: mean(rs.map((r) => r.maxActivation)),
  };
}

// Para ablasjon: same teiknvokabular, same frø, same felles startvekter,
// same minibatchar og same held-out utdrag. Berre trigramtabellen skil sidene.
export function runNgramAblation(o: NgramAblationOptions): NgramAblationResult {
  validateOptions(o);
  const ngram = { ...(o.ngram ?? NGRAM_DEFAULT) };
  const evalBatches = o.evalBatches ?? 24;
  if (!Number.isInteger(evalBatches) || evalBatches < 1)
    throw new RangeError("evalBatches must be positive");

  const tokenizer = buildTokenizer(`${o.trainText}\n${o.holdoutText}`);
  const trainData = tokenizer.encode(o.trainText);
  const holdoutData = tokenizer.encode(o.holdoutText);
  const trainCharacters = Array.from(o.trainText).length;
  const holdoutCharacters = Array.from(o.holdoutText).length;
  if (trainData.length !== trainCharacters || holdoutData.length !== holdoutCharacters)
    throw new Error("character tokenizer invariant failed");

  const runs: AblationRun[] = [];
  let maxInitDiff = 0;
  for (let pair = 0; pair < o.seeds.length; pair++) {
    const seed = o.seeds[pair];
    const common = { ...o.baseConfig, vocab: tokenizer.vocab };
    const baseline = new Transformer(common, mulberry32(seed));
    const trigram = new Transformer({ ...common, ngram }, mulberry32(seed));
    maxInitDiff = Math.max(maxInitDiff, sharedInitDiff(baseline, trigram));

    // Alterner kven som køyrer først, så JIT-oppvarming ikkje systematisk blir
    // bokført som ein kostnad eller gevinst for same variant.
    const order: [AblationVariant, Transformer][] =
      pair % 2 === 0
        ? [
            ["baseline", baseline],
            ["trigram", trigram],
          ]
        : [
            ["trigram", trigram],
            ["baseline", baseline],
          ];
    for (const [variant, model] of order)
      runs.push(runOne(variant, seed, model, trainData, holdoutData, o, evalBatches));
  }

  const baseline = summarize("baseline", runs);
  const trigram = summarize("trigram", runs);
  return {
    tokenizer: {
      vocab: tokenizer.vocab,
      trainCharacters,
      trainTokens: trainData.length,
      holdoutCharacters,
      holdoutTokens: holdoutData.length,
      oneCharacterPerToken: true,
    },
    config: {
      base: { ...o.baseConfig },
      ngram,
      steps: o.steps,
      batchSize: o.batchSize,
      learningRate: o.learningRate,
      seeds: o.seeds.slice(),
      evalBatches,
    },
    sharedInitMaxDiff: maxInitDiff,
    runs,
    baseline,
    trigram,
    delta: {
      params: trigram.params - baseline.params,
      paramsPct: pct(trigram.params, baseline.params),
      heldoutLoss: trigram.heldoutLoss - baseline.heldoutLoss,
      heldoutLossPct: pct(trigram.heldoutLoss, baseline.heldoutLoss),
      msPerStepPct: pct(trigram.msPerStep, baseline.msPerStep),
      gradP99Pct: pct(trigram.gradP99, baseline.gradP99),
      clipRate: trigram.clipRate - baseline.clipRate,
      maxActivationPct: pct(trigram.maxActivation, baseline.maxActivation),
    },
  };
}
