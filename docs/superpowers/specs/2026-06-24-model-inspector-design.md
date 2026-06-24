# Model inspector — "Se inni modellen" — design spec

**Date:** 2026-06-24
**App:** Norsk LLM trainer (in-browser, character-level transformer with hand-written autograd)
**Status:** Approved (design); pending spec review → implementation plan

## Overview

Add an interactive **"Se inni modellen" (Look inside the model)** section to the existing
single-page app. It makes the two things a beginner can't currently see — **what the model
attends to** and **what it actually predicts** — visible, and links them into one lesson:

> *Click a character → see what that position looks back at (attention) → see what it therefore
> predicts comes next (the probability distribution).*

This extends the app's "everything is real" ethos to inference: it renders the **real** post-softmax
attention weights and the **real** next-character probability distribution from a single forward
pass of the same model trained in the Training section. Nothing is faked or re-derived for display.

The primary audience is **curious beginners** with little/no ML background. The educational payoff
is the *link* between attention and prediction, which no existing section shows.

## Goals

- Surface the model's **next-character probability distribution** as sorted bars — the literal
  output a beginner never sees behind generated text.
- Surface **attention** as a per-layer, per-head heatmap (query rows × key columns), with the
  causal upper-triangle left blank as its own teaching moment ("it can only look backward").
- **Link** the two: clicking a character selects a query position that drives both the highlighted
  attention row and the probability bars.
- Provide an honest micro-feedback loop ("Fasit"): show the *actual* next character with a ✓/✗
  against the model's top guess.
- Work on the **same model** as the rest of the page (untrained → trained), so a learner sees flat
  bars / diffuse attention before training and sharpening / focusing after.

## Non-goals (YAGNI)

- No live mid-training streaming of the panels (that was approach C; out of scope). Panels refresh
  when a training run completes, not on every step.
- No "be the model" prediction game (deferred — a separate future feature on top of `rowProbs`).
- No embedding map / 2D projection.
- No gradient/backward visualization — inspection is forward-only.
- No persistence of the inspector's input text across reloads.
- No temperature/top-k reshaping in the bars (show the raw softmax distribution; temperature lives
  in the Try-the-model section).

## Decisions (locked during brainstorming)

1. **Audience:** curious beginners. The gap addressed is that today the model's internals are
   invisible and the learner is passive.
2. **Structure:** one **unified inspector section** (chosen over extending existing sections, and
   over the live-during-training variant). One input drives both panels from a single forward pass.
3. **Interaction:** **linked** — click any character to select a query position that drives both
   the attention row and the probability bars (chosen over two static, decoupled panels).
4. **Placement:** between **Training (§3)** and **Try the model (§4)**, renumbering later sections.
   Flow: train it → peek inside its head on one prediction → let it generate (the same prediction,
   looped).
5. **Feedback line ("Fasit"):** keep — show the real next character with ✓/✗ vs the model's top
   guess.

## Architecture & components

### 1. ML core (`src/lib/ml.ts`) — surface what is already computed

Two additions, both pure and unit-testable; neither alters the training/generate path.

```ts
// Softmax of one logits row → a probability distribution over the vocabulary.
export function rowProbs(logits: Tensor, pos: number): Float32Array

// One captured attention matrix (post-softmax). row = query position, col = key position.
export interface AttnView {
  layer: number;
  head: number;
  T: number;
  weights: Float32Array; // length T*T, row-major (row = query, col = key)
}

// On Transformer: mirrors forward(), but also records every head's attention.
inspect(ids: number[]): { logits: Tensor; attn: AttnView[] }
```

**Implementation:** thread an **optional sink** (`AttnView[] | undefined`) through the private
`attention()` → `blockForward()` → `forward()` methods. When a sink is passed, each head pushes a
**copy** of its `softmaxRow(scores).d` as an `AttnView`. When the sink is absent — the training and
generation path — behavior is byte-for-byte unchanged. `inspect()` calls the forward logic with a
fresh sink, then returns `{ logits, attn }`. `attn.length === nLayer * nHead`; each `weights` is
length `T*T`. No backward pass is run for inspection.

This avoids duplicating forward logic and guarantees the hot training path is untouched.

### 2. New component (`src/components/Inspector.tsx`)

Presentational, in the style of `LossChart`/`Architecture` (self-drawn SVG / Tailwind divs, no
chart library).

- **Props:** `{ model: Transformer; tokenizer; step: number; s: Strings["inspect"] }`.
- **Local UI state:** `inputText`, `selectedLayer`, `selectedHead`, `selectedPos`.
- **Compute:** a `useMemo` keyed on `(inputText, step)` runs `model.inspect(encode(inputText))`
  once. Changing layer / head / position is a pure re-render off that memo — no recompute. Keying
  on `step` means the panels refresh after a training run completes.
- **Input handling:** cap to `model.seqLen` by slicing the tail (mirroring `sampleTokens`); empty
  input falls back to `[0]`; debounce text edits ~150 ms.
- **Renders:**
  - A row of **clickable character chips** for the input; clicking position *i* sets `selectedPos`.
    Default `selectedPos` = last character.
  - **Attention panel:** grid heatmap, rows = query positions, cols = key positions, cell opacity =
    weight; upper triangle (key > query) left blank; selected query row highlighted. **Layer** and
    **head** selectors above it (segmented buttons / dropdowns).
  - **Probability panel:** horizontal bars of `rowProbs(logits, selectedPos)` over the vocab,
    sorted descending, argmax highlighted, characters labeled (space shown as `' '`).
  - **Fasit line:** the actual next character at `selectedPos + 1` in the input (if any), with ✓/✗
    against the model's top guess.

### 3. App integration (`src/App.tsx`)

- One new `<Section step={4}>` between Training and Try-the-model, wrapping
  `<Inspector model={engineRef.current.model} tokenizer={…} step={step} s={s.inspect} />`.
- Renumber the subsequent sections (Try-the-model, RLHF, Add-your-own-text) `step` +1.
- No changes to state management — the model is read from `engineRef.current.model`; `step` already
  exists as React state.

### 4. i18n (`src/lib/i18n.ts`)

Add an `inspect` block to the `Strings` interface and to **both** the `bm` and `nn` bundles:
section title, intro, "click a character" hint, layer/head labels, probability heading, Fasit/✓/✗
strings, untrained hint, and a per-language **default sample sentence**. The existing `i18n-parity`
test then enforces both bundles share the same shape.

## Data flow

```
inputText ──encode──▶ ids (capped to seqLen)
   │
   └─(useMemo on inputText, step)─▶ model.inspect(ids) ─▶ { logits, attn }
                                                              │        │
   selectedLayer/head ──────────────────────────────────────┘        │
        └─▶ pick attn view ─▶ heatmap (row = selectedPos highlighted) │
   selectedPos ───────────────────────────────────────────────────────┘
        └─▶ rowProbs(logits, selectedPos) ─▶ sorted bars + Fasit(next char)
```

## Error handling & edge cases

- **No model yet / empty corpus:** the section shows the untrained hint instead of computing.
- **Untrained model:** functions normally (flat bars, diffuse attention) — an intended teaching
  state, with a gentle "train first" nudge.
- **Single-character input:** degenerate 1×1 heatmap; valid.
- **Input shortened:** clamp `selectedPos` to the new length.
- **Selected position is the last char:** no real "next" character → Fasit shows the top guess
  without a ✓/✗ verdict.

## Testing (`test/`)

New Node tests for the pure logic (no React), consistent with the existing suite:

- **`rowProbs`** — non-negative, sums to ~1, matches a hand-computed softmax of a known row.
- **`inspect` shape & causality** — `attn.length === nLayer*nHead`; every query row sums to ~1 over
  allowed keys and is exactly 0 for keys > query (causal mask holds).
- **`inspect` consistency** — `inspect(ids).logits` equals `forward(ids)` logits (the sink does not
  change the math).
- **`i18n-parity`** (existing) automatically covers the new `inspect` strings.

## Effort estimate

Roughly one focused build. The ML additions are small (a softmax helper + an optional capture
sink); the bulk is the linked-panel UI and the bilingual strings. Rendering reuses existing
SVG/Tailwind idioms.
