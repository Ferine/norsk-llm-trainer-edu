import { Transformer, generate, sampleTokens, mulberry32 } from "./dist/ml.js";
import { buildTokenizer, corpus } from "./dist/corpus.js";
import assert from "node:assert/strict";

const tok = buildTokenizer(corpus);
const cfg = { vocab: tok.vocab, dim: 16, nLayer: 1, nHead: 2, seqLen: 16, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(5));
const prompt = "Det var";

// greedy: deterministic, and generate is exactly prompt + decode(sampleTokens.contIds)
const og = { temperature: 0, topK: 5, length: 10 };
const g = generate(m, tok.decode, tok.encode, prompt, og, mulberry32(1));
const st = sampleTokens(m, tok.encode, prompt, og, mulberry32(1));
assert.equal(g, prompt + tok.decode(st.contIds));
assert.ok(g.startsWith(prompt));
assert.equal(st.contIds.length, 10);

// sampling: identical seed yields identical result through both entry points
const os = { temperature: 0.9, topK: 5, length: 12 };
const g2 = generate(m, tok.decode, tok.encode, prompt, os, mulberry32(9));
const st2 = sampleTokens(m, tok.encode, prompt, os, mulberry32(9));
assert.equal(g2, prompt + tok.decode(st2.contIds));
console.log("generate-parity: PASS");
