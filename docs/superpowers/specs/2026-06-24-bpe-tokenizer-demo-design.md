# BPE tokenizer demo — "Fra tegn til ord-biter" — design spec

**Date:** 2026-06-24
**App:** Norsk LLM trainer (in-browser, character-level transformer with hand-written autograd)
**Status:** Approved (design); pending spec review → implementation plan

## Overview

Add an interactive **"Fra tegn til ord-biter" (From characters to subwords)** section that teaches
**Byte-Pair Encoding (BPE)** — the subword tokenization real LLMs use — by letting the learner run
the algorithm live on the app's own Norwegian corpus and watch a subword vocabulary emerge.

The current app tokenizes at the **character** level (`buildTokenizer` in `src/lib/corpus.ts`).
That is the single biggest fidelity gap versus production models, and also one of the richest
teaching opportunities: tokenization is a famous source of LLM confusion (e.g. why models miscount
the letters in a word). This section closes the gap pedagogically.

It is a **standalone visualization**: it does NOT change the model's tokenizer or training. The
model keeps training on character-level tokens. The BPE demo is self-contained, pure, and isolated.

The primary audience is **curious beginners**. The payoff is seeing the mechanism — *repeatedly
merge the most frequent adjacent pair* — produce real Norwegian morphology, and understanding the
core tradeoff (vocabulary grows, sequence length shrinks).

## Goals

- Show BPE **learning its merges** step by step on the live corpus: each step merges the
  **most frequent** adjacent symbol pair, and the learner can advance one merge at a time or scrub
  to any point.
- Make the central tradeoff physical: a **sample sentence re-tokenizes live**, its token count
  visibly dropping from all-characters toward fewer subwords as merges accumulate.
- Show **why** each pair is chosen by displaying its frequency and its top runners-up
  ("konkurrenter").
- Surface the growing **merge-rule list** (the same kind of ordered rules GPT-2 ships in
  `merges.txt`) and the **vocabulary tally** climbing by exactly one per merge.
- Tie it to reality with one honest **payoff callout**: once a word collapses into a single token,
  the model sees an opaque symbol, not letters — why LLMs miscount letters.
- Be authentically bilingual: because merges are learned from the active corpus, **Bokmål and
  Nynorsk produce different subwords** with no extra work.

## Non-goals (YAGNI)

- Does NOT replace or augment the model's tokenizer; no change to training, model size, generation,
  or the inspector.
- No byte-level / regex pre-tokenizer (GPT-2 style); whitespace word-splitting is sufficient and
  clearer for a demo.
- No end-of-word marker; pairs never cross word boundaries (words are encoded independently).
- No auto-play timer (the control is step + scrubber).
- No persistence of the demo state across reloads.
- No letter-counting interactive game — the "strawberry" insight is a single static callout.

## Decisions (locked during brainstorming)

1. **Scope:** standalone visualization; the model keeps its character tokenizer.
2. **Control:** **step + scrubber** — a "Slå sammen neste par" button to advance one merge, plus a
   slider to scrub 0→N. Merges are precomputed once, so both are pure renders.
3. **Placement:** a **new section** "Fra tegn til ord-biter" immediately after §1 (tokenization),
   renumbering the rest (architecture→§3, training→§4, inspector→§5, try→§6, RLHF→§7,
   add-text→§8).
4. **Keep the "konkurrenter" panel** — showing the chosen pair's frequency and top runners-up is
   the clearest way to convey the "most frequent pair wins" rule.
5. **Tie-break:** among max-count pairs, pick the **lexicographically smallest** `(a, then b)` —
   deterministic, for stable tests and UI.
6. **`NUM_MERGES = 80`**, with **early stop** when the best pair count drops below 2.

## Architecture & components

### 1. New pure module (`src/lib/bpe.ts`)

The algorithm, fully isolated and unit-testable. No React, no DOM.

```ts
export interface BpeMerge {
  a: string;        // left symbol
  b: string;        // right symbol
  merged: string;   // a + b
  count: number;    // frequency when chosen
  rank: number;     // 0-based learn order
  rivals: { pair: string; count: number }[]; // top runners-up this step (<= 2)
}
export interface BpeResult {
  baseVocab: string[]; // sorted unique non-space characters in the corpus
  merges: BpeMerge[];  // in learned order
}

// Learn up to numMerges merges from text. Stops early when the best pair count < 2.
export function learnBpe(text: string, numMerges: number): BpeResult;

// Encode one word into subword tokens using the first k merges (greedy, in rank order).
export function encodeWord(word: string, merges: BpeMerge[], k: number): string[];

// Encode a sentence: split into words on whitespace, encode each with the first k merges,
// concatenate the per-word subwords into one token list.
export function tokenizeSentence(sentence: string, merges: BpeMerge[], k: number): string[];
```

**`learnBpe` algorithm:**
1. Split `text` on whitespace into words; build a `word → frequency` map. Empty words dropped.
2. Represent each unique word as a symbol array via `Array.from` (correct Unicode for å/ø/æ).
   `baseVocab` = sorted unique characters across all words.
3. Repeat up to `numMerges` times:
   - Count every adjacent symbol pair, weighting each word's pairs by its frequency.
   - If there are no pairs, or the max count < 2, **stop**.
   - Choose the max-count pair; break ties by lexicographically smallest `(a, b)`.
   - Record a `BpeMerge` with `count`, `rank`, and the next ≤2 distinct pairs by count as `rivals`.
   - Replace that pair with the merged symbol greedily (left-to-right) in every word's symbol array.
4. Return `{ baseVocab, merges }`. `merges.length <= numMerges`.

**`encodeWord`:** start from `Array.from(word)`; apply `merges[0..k-1]` in rank order, each as a
greedy left-to-right replacement of its `(a, b)` pair. Returns the resulting symbol list. Works for
words/characters not in the training corpus (they simply have fewer applicable merges).

### 2. New component (`src/components/BpeLab.tsx`)

Presentational, in the style of `LossChart`/`Architecture`/`Inspector` (self-drawn Tailwind divs,
no chart library, no unit test — consistent with the other components).

- **Props:** `{ corpus: string; sampleSentence: string; s: Strings["bpe"] }`.
- **Compute:** a `useMemo` keyed on `corpus` runs `learnBpe(corpus, NUM_MERGES)` once.
- **Local state:** `applied` (number of merges applied, 0..merges.length), default 0.
- **Reset on language switch:** a `useEffect` on `corpus` sets `applied = 0`; render also clamps
  `appliedClamped = Math.min(applied, merges.length)` to stay valid if `merges` shrinks.
- **Controls:** "Slå sammen neste par" (`applied = min(applied + 1, merges.length)`), "Nullstill"
  (`applied = 0`), and a slider (`min=0`, `max=merges.length`).
- **Derived (pure from `appliedClamped`):**
  - `currentMerge = appliedClamped > 0 ? merges[appliedClamped - 1] : null`
  - `rules = merges.slice(0, appliedClamped)`
  - `vocabSize = baseVocab.length + appliedClamped`
  - `tokens = tokenizeSentence(sampleSentence, merges, appliedClamped)`;
    `baseCount = tokenizeSentence(sampleSentence, merges, 0).length`
- **Renders:** the controls, "Denne sammenslåingen" panel (current pair, count, rivals), "Reglene så
  langt" list, the re-tokenized sentence chips (multi-character/merged tokens highlighted) with the
  current vs base token count, the vocabulary tally, and the static payoff callout.

### 3. App integration (`src/App.tsx`)

- Insert a new `<Section id="bpe" step={2}>` immediately after the data/tokenization section,
  wrapping `<BpeLab corpus={activeCorpus} sampleSentence={sampleSentence} s={s.bpe} />` in a
  `<Card>`. (`activeCorpus` and `sampleSentence` already exist in `App.tsx`.)
- Renumber the later `<Section>` badges: architecture 2→3, training 3→4, inspector 4→5, chat 5→6,
  RLHF 6→7, add-text 7→8. Change ONLY `<Section step={N}>` props; never the `<input type="range">`
  `step` values.

### 4. i18n (`src/lib/i18n.ts`)

Add a `bpe` block to the `Strings` interface and to **both** the `bm` and `nn` bundles: title,
intro, the two button labels, the merge-count label, the "this merge" strings (pair, found-N-times,
rivals label), the rules heading, the sentence heading (with current/base token counts), the
vocabulary line, and the payoff callout. The existing `i18n-parity` test enforces both bundles share
the same key shape and the same string/function typing per key.

## Data flow

```
corpus ──(useMemo on corpus)──▶ learnBpe(corpus, NUM_MERGES) ──▶ { baseVocab, merges }
   │
applied (state) ──clamp──▶ appliedClamped
   ├─▶ merges[appliedClamped-1]                 ─▶ "Denne sammenslåingen" (+ rivals)
   ├─▶ merges.slice(0, appliedClamped)          ─▶ "Reglene så langt"
   ├─▶ baseVocab.length + appliedClamped         ─▶ vocabulary tally
   └─▶ tokenizeSentence(sample, merges, applied) ─▶ sentence chips + token count (vs base)
```

## Error handling & edge cases

- **Empty corpus / corpus with no repeated pairs:** `learnBpe` returns `merges = []`; the component
  shows 0 merges, slider range `[0, 0]`, sentence stays at characters. No crash.
- **Single-character words:** contribute no pairs; ignored naturally.
- **Language switch (corpus change):** merges recompute; `applied` resets to 0 and is clamped.
- **Words in the sample sentence not present in the corpus:** `encodeWord` still applies whatever
  merges fit their characters; safe.
- **`applied` at 0:** "Denne sammenslåingen" panel shows an empty/placeholder state (no current
  merge), sentence is all characters.

## Testing (`test/bpe.test.mjs`, Node, pure)

- **Most-frequent-first:** on a tiny corpus where one pair dominates, `merges[0]` is that pair.
- **Determinism:** `learnBpe(text, n)` run twice is `deepEqual`.
- **Vocab growth:** `merged === a + b`; `rank` values are `0..merges.length-1` sequential; vocabulary
  size grows by exactly 1 per merge.
- **Tie-break:** a constructed tie yields the lexicographically smaller pair.
- **`encodeWord`:** `k=0` returns the character list; applying all merges returns the expected
  subwords for a known word; token count is **monotonically non-increasing** as `k` grows.
- **Early stop / edge:** a corpus with no repeated pairs yields `merges = []`.
- **`i18n-parity`** (existing) automatically covers the new `bpe` strings.

## Effort estimate

Roughly one focused build, the same shape as the model-inspector feature: a compact pure library
(`bpe.ts`), a moderate presentational component (`BpeLab.tsx`), bilingual strings, a new section with
a mechanical renumber, and a Node test file.
