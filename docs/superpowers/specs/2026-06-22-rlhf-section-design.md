# RLHF section — design spec

**Date:** 2026-06-22
**App:** Nynorsk LLM trainer (in-browser, character-level transformer with hand-written autograd)
**Status:** Approved (design); pending spec review → implementation plan

## Overview

Add an interactive **RLHF (Reinforcement Learning from Human Feedback)** section to the
existing single-page app. It teaches preference alignment using **DPO (Direct Preference
Optimization)**: the visitor is shown two model-generated continuations of a prompt, picks the
better one, and the policy is fine-tuned directly toward the preferred continuation, anchored to a
frozen reference copy via a KL term (implicit in the DPO objective).

This keeps the app's "everything is real" ethos: real per-sequence log-probabilities, real
gradients through the existing autograd engine, real optimizer steps — no mock-ups.

## Goals

- Demonstrate the *concept* of preference-based alignment with a faithful, stable, in-browser
  algorithm (DPO).
- Operate on the **same model** trained in Section 3 ("Trening") so the page tells one continuous
  story: random weights → supervised training (SFT) → preference-aligned.
- Let the visitor feel the learning: a **live nudge** on each preference pick, plus a **"train
  more"** button for a longer pass over the collected preferences.
- Show meaningful live metrics (DPO loss, implicit reward margin, win-rate, #preferences).
- Be reversible: a reset that reverts DPO changes (restores the frozen reference) without
  destroying the SFT progress.

## Non-goals (YAGNI)

- No separate reward model (DPO does not need one).
- No REINFORCE/PPO RL loop.
- No best-of-N inference reranking.
- No persistence/export of preferences across reloads.
- No multi-prompt batched campaigns beyond the single running preference buffer.

## Decisions (locked during brainstorming)

1. **Mechanism:** DPO-style preference fine-tuning (chosen over reward-model+REINFORCE and
   reward-model+best-of-N) — real policy optimization, stable on a tiny model, reuses existing
   autograd, no RL instability.
2. **Update timing:** Both — a live burst of DPO steps on each preference pick **and** a separate
   "train more on the buffer" button for a longer pass.
3. **Base model:** Reuse the Section 3 model. Snapshot a frozen reference copy when preference
   tuning starts; DPO updates the live model, so the chat section afterward reflects the aligned
   model.

## Section 1 — Math additions to `src/lib/ml.ts`

### Per-sequence log-probability

For next-token prediction over a full sequence `ids` of length `T`, `logits[t]` predicts
`ids[t+1]`. A continuation occupying full-sequence indices `[P, T-1]` is therefore scored by
logit rows `[P-1, T-2]` against targets `ids[P..T-1]`.

```ts
// Autograd: Σ log softmax(logits[r0+i])[targets[i]] for i in [0, targets.length).
// Backward: d(logp)/dlogit = (onehot(target) − softmax). Mirrors crossEntropyLoss internals.
export function seqLogProb(logits: Tensor, r0: number, targets: number[]): Tensor; // → [1,1]

// Numeric-only version for the frozen reference (no graph, returns a number).
export function seqLogProbValue(logits: Tensor, r0: number, targets: number[]): number;
```

### DPO loss

```ts
// z = β·((logπθ_w − logπref_w) − (logπθ_l − logπref_l))
// loss = −log σ(z) = softplus(−z)   (numerically stable form)
// dloss/dz = −σ(−z);  chains to lpW (×β) and lpL (×−β).
export function dpoLoss(lpW: Tensor, lpL: Tensor, refW: number, refL: number, beta: number): Tensor;
```

`lpW`/`lpL` are policy log-prob scalars (autograd, from `seqLogProb`). `refW`/`refL` are reference
log-probs (plain numbers, from `seqLogProbValue`). `β` default **0.1**.

### Reference snapshot

```ts
// Deep-copy params into a new Transformer with the same cfg → frozen reference.
// new Transformer(src.cfg, mulberry32(0)); then dst.params[i].d.set(src.params[i].d).
export function cloneTransformer(src: Transformer): Transformer;
```

Param array order is deterministic for a given cfg, so an index-wise copy is sound. The reference
is used only for forward passes; its grads are never read.

### DPO step

```ts
export interface PrefPair { promptIds: number[]; chosenIds: number[]; rejectedIds: number[]; }

// One DPO update over a sampled minibatch of `batch` pairs.
// Zero grads → accumulate dpoLoss over the minibatch → average → clipGradNorm(1.0) → opt.step().
export function dpoStep(
  policy: Transformer, reference: Transformer, opt: Adam,
  pairs: PrefPair[], batch: number, beta: number, rng: () => number
): { loss: number; margin: number; winRate: number };
```

- **margin** = mean implicit-reward gap `(lpW−refW) − (lpL−refL)` over the minibatch (no β factor).
- **winRate** = fraction of minibatch pairs with margin > 0.

### Autograd notes (avoid cross-contamination)

- Reference log-probs enter `dpoLoss` as plain numbers (`refW`/`refL`), so the reference forward
  graph is **disconnected** from `backward(loss)` — only the policy subgraphs (ancestors of `lpW`
  and `lpL`) are traversed. The reference's params therefore never receive gradients, and the DPO
  `Adam` only holds `policy.params`.
- `lpW` and `lpL` come from two separate policy forward passes that **share the same policy param
  leaf tensors**. `backward` accumulates gradients from both sequences into those shared leaves —
  the same fan-out accumulation already verified for residual connections in the bug review. Grads
  must be zeroed once per `dpoStep` (handled by `opt.zeroGrad()` at the top of the step), not
  between the two forwards.

### Sequence capping rule

Each pair's full sequence is `prompt + continuation`, capped to `model.seqLen`:

1. Ensure `promptLen ≥ 1` (if the prompt encodes to empty, seed with token `0`, matching
   `generate`).
2. Ensure `contLen ≥ 1`.
3. If `promptLen + contLen > seqLen`: truncate the **prompt from the left** first
   (`promptLen = max(1, seqLen − contLen)`); if `contLen ≥ seqLen`, also truncate the continuation
   to `seqLen − 1` and keep `promptLen = 1`.

Scored rows: `r0 = promptLen − 1`, `targets = fullSeq.slice(promptLen)` (length `contLen`),
so rows used are `[promptLen−1, T−2]` — always in range.

### Sampling refactor (to obtain continuation token ids)

```ts
// Shared sampling core used by both generate() and the preference arena.
export function sampleTokens(
  model: Transformer, encode: (s: string) => number[],
  prompt: string, opts: SampleOpts, rng: () => number
): { promptIds: number[]; contIds: number[] };

// generate(...) becomes: prompt + decode(sampleTokens(...).contIds)
```

This is a behavior-preserving refactor of the current `generate` logic (greedy + top-k sampling,
sliding context window). The bug-review verification already covers `generate`; the refactor must
keep that path identical.

## Section 2 — UI section & interaction flow (`src/App.tsx` + new component)

**Placement & numbering:** insert a new **Section step 5 "RLHF – lær modellen kva vi
føretrekkjer"** immediately after the chat section (step 4). The amber "honest about what this is"
note stays after it, and **"Legg til eigen tekst" renumbers from step 5 → step 6**.

**Modularization:** to keep `App.tsx` (already ~720 lines) focused, the RLHF orchestration lives in
a dedicated hook `src/lib/useRlhf.ts` (state, refs, handlers) and the UI in a presentational
component `src/components/Rlhf.tsx`. `App` wires the hook to the component and passes the current
engine.

### Flow

1. **"Start preferanse-trening"** → `reference = cloneTransformer(model)`, create a dedicated DPO
   `Adam` (lr ≈ **1e-4**), enable the arena, set `rlhfStarted = true`. If the model is undertrained
   (no SFT steps yet) show a non-blocking hint: *"Tren modellen først i steg 3 for meiningsfulle
   svar."*
2. **Prompt field** + the existing example chips, and a small **"kreativitet"** slider (sampling
   temperature for pair diversity, default **1.0**).
3. **"Generer eit par"** → two continuation cards **A** and **B** sampled from the same prompt. If A
   and B decode identically, auto-resample (capped retries) so the choice is meaningful.
4. **"👍 A er betre" / "👍 B er betre"** (plus **"Hopp over"** for ties) → build
   `(chosen, rejected)`, push to the buffer, run a **live burst (~5 `dpoStep` calls,
   minibatch = min(buffer, 4))**, update metrics, auto-generate the next pair.
5. **"Tren meir på preferansane"** → a chunked DPO loop (same `setTimeout` chunk pattern as the SFT
   loop, ~60 steps, minibatch 4) over the whole buffer, with a live curve; stoppable.
6. **Metrics panel:** DPO loss via the existing `LossChart`, plus reward-margin, win-rate, and
   #preferansar samla as badges. Optional lightweight **"Samanlikn før/no"**: one sample from the
   frozen `reference` vs one from the current policy for the current prompt.
7. **"Nullstill preferanse-trening"** → copy the reference params back into the policy (reverts DPO,
   keeps SFT), clear buffer + metrics, keep `rlhfStarted = true`.

Because DPO updates the live model, the existing **"Prøv modellen" chat section reflects the aligned
model** after tuning.

## Section 3 — Integration, state, and edge cases

### State / refs (in `useRlhf`)

- `referenceRef: Transformer | null` — frozen reference snapshot.
- `dpoOptRef: Adam | null` — DPO optimizer over `policy.params`.
- `prefBufferRef: PrefPair[]` — collected preferences.
- `dpoLossesRef` + `dpoLosses` state — for the chart.
- `pairA` / `pairB` state — `{ text, promptIds, contIds }`.
- `rlhfStarted`, `dpoRunning` flags; metric state `{ margin, winRate, count }`.
- `pairRngRef = mulberry32(...)` — dedicated RNG stream for pair sampling (reproducible; does not
  perturb the SFT/chat RNGs).
- `dpoTimerRef` — chunked-loop timer; cleared on unmount.

### Mutual exclusion

- RLHF controls disabled while base training `running`.
- Base "Start trening" disabled while `dpoRunning`.
- Reuse the existing ref-guard + timer pattern from the SFT loop.

### Invalidation

`buildEngine` (preset change, reset, rebuild-with-extra-text) changes the model identity, so it must
fully reset RLHF state: clear `referenceRef`, `dpoOptRef`, `prefBufferRef`, metrics, losses, pairs,
and set `rlhfStarted = false`. This composes with the existing rebuild logic.

### Edge cases

- Empty prompt → seed with token `0` (matches `generate`).
- Identical A/B → resample (capped retries); if still identical, allow "Hopp over".
- Buffer smaller than minibatch → use the whole buffer.
- Sequence longer than `seqLen` → capping rule above.
- Numerical stability: stable softplus and max-subtracted softmax in `seqLogProb`/`dpoLoss`.

## Files changed

- **`src/lib/ml.ts`** — add `seqLogProb`, `seqLogProbValue`, `dpoLoss`, `cloneTransformer`,
  `sampleTokens`, `dpoStep`, `PrefPair`; refactor `generate` onto `sampleTokens`.
- **`src/lib/useRlhf.ts`** (new) — RLHF state/refs/handlers hook.
- **`src/components/Rlhf.tsx`** (new) — presentational RLHF section UI.
- **`src/App.tsx`** — render the RLHF section (step 5), renumber "eigen tekst" to step 6, invalidate
  RLHF state inside `buildEngine`, wire mutual-exclusion flags.

## Testing

Mirror the verification approach used in the bug review (transpile with `tsc`, run under Node):

1. **Gradient checks** — numerically verify `seqLogProb` (autograd vs central differences) and
   `dpoLoss` (grad into `lpW`/`lpL`), using fixed reduction weights and Float32-aware tolerances.
2. **Smoke test** — SFT-train ~200 steps on the corpus, snapshot the reference, build synthetic
   preferences where `chosen` is a real corpus continuation and `rejected` is random tokens, run
   ~50 `dpoStep`s, and assert: win-rate and reward-margin increase, and no NaN/Inf appears in
   params.
3. **`generate` parity** — confirm the `sampleTokens` refactor leaves `generate` output identical
   for fixed seed/prompt/opts.
4. **App** — `pnpm typecheck` and `pnpm build` pass.

## Success criteria

- `pnpm typecheck` and `pnpm build` pass.
- New-op gradient checks pass.
- Smoke test shows DPO improving margin/win-rate on synthetic preferences with no NaNs.
- Manual: start RLHF → label a few pairs → loss/margin/win-rate move → chat reflects the change →
  reset reverts to the reference; switching preset/reset clears RLHF state cleanly.
