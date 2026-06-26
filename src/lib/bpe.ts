// Byte-Pair Encoding (BPE) — a small, dependency-free implementation for the
// "Fra tegn til ord-biter" teaching section. Pure: no React, no DOM.
// STANDALONE: this does NOT tokenize for the model — the model uses the
// character tokenizer in corpus.ts. This module only powers the BPE demo.

export interface BpeMerge {
  a: string; // left symbol
  b: string; // right symbol
  merged: string; // a + b
  count: number; // how many times the pair occurred when it was chosen
  rank: number; // 0-based order in which this merge was learned
  rivals: { pair: string; count: number }[]; // up to 2 runner-up pairs this step
}

export interface BpeResult {
  baseVocab: string[]; // sorted unique characters across the corpus words
  merges: BpeMerge[]; // in the order they were learned
}

// NUL separates the two symbols of a pair key; corpus text never contains NUL.
const SEP = "\u0000";

// Split text into words on whitespace → a frequency map of words.
function wordFreqs(text: string): Map<string, number> {
  const freqs = new Map<string, number>();
  for (const w of text.split(/\s+/)) {
    if (w.length === 0) continue;
    freqs.set(w, (freqs.get(w) ?? 0) + 1);
  }
  return freqs;
}

// Count every adjacent symbol pair across all words, weighted by word frequency.
function countPairs(words: { syms: string[]; freq: number }[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const { syms, freq } of words) {
    for (let i = 0; i + 1 < syms.length; i++) {
      const key = syms[i] + SEP + syms[i + 1];
      counts.set(key, (counts.get(key) ?? 0) + freq);
    }
  }
  return counts;
}

// Replace each adjacent (a, b) with a+b, greedily left-to-right.
function mergeSymbols(syms: string[], a: string, b: string): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < syms.length) {
    if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
      out.push(a + b);
      i += 2;
    } else {
      out.push(syms[i]);
      i += 1;
    }
  }
  return out;
}

// Learn up to numMerges merges from text. Deterministic; stops early when the
// most frequent remaining pair occurs fewer than 2 times.
export function learnBpe(text: string, numMerges: number): BpeResult {
  const words = Array.from(wordFreqs(text), ([w, freq]) => ({ syms: Array.from(w), freq }));

  const vocabSet = new Set<string>();
  for (const { syms } of words) for (const c of syms) vocabSet.add(c);
  const baseVocab = Array.from(vocabSet).sort();

  const merges: BpeMerge[] = [];
  for (let rank = 0; rank < numMerges; rank++) {
    const counts = countPairs(words);
    if (counts.size === 0) break;

    // total order: count desc, then a asc, then b asc → deterministic choice.
    const ranked = Array.from(counts, ([key, count]) => {
      const sep = key.indexOf(SEP);
      return { a: key.slice(0, sep), b: key.slice(sep + 1), count };
    }).sort(
      (x, y) =>
        y.count - x.count ||
        (x.a < y.a ? -1 : x.a > y.a ? 1 : x.b < y.b ? -1 : x.b > y.b ? 1 : 0)
    );

    const best = ranked[0];
    if (best.count < 2) break;

    const rivals = ranked.slice(1, 3).map((r) => ({ pair: r.a + r.b, count: r.count }));
    merges.push({ a: best.a, b: best.b, merged: best.a + best.b, count: best.count, rank, rivals });

    for (const word of words) word.syms = mergeSymbols(word.syms, best.a, best.b);
  }

  return { baseVocab, merges };
}

// Encode one word into subword tokens using the first k merges (greedy, in rank order).
export function encodeWord(word: string, merges: BpeMerge[], k: number): string[] {
  let syms = Array.from(word);
  const n = Math.min(k, merges.length);
  for (let i = 0; i < n; i++) syms = mergeSymbols(syms, merges[i].a, merges[i].b);
  return syms;
}

// Encode a sentence: split on whitespace, encode each word with the first k merges,
// concatenate the per-word subwords into one token list.
export function tokenizeSentence(sentence: string, merges: BpeMerge[], k: number): string[] {
  const out: string[] = [];
  for (const w of sentence.split(/\s+/)) {
    if (w.length === 0) continue;
    for (const tok of encodeWord(w, merges, k)) out.push(tok);
  }
  return out;
}
