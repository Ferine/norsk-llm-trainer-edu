import { Transformer, generate, generateDetailed, sampleTokens, mulberry32 } from "./dist/ml.js";
import { buildTokenizer, corpus } from "./dist/corpus.js";
import assert from "node:assert/strict";

const tok = buildTokenizer(corpus);
const cfg = { vocab: tok.vocab, dim: 16, nLayer: 1, nHead: 2, seqLen: 16, ffnMult: 2 };
const m = new Transformer(cfg, mulberry32(5));
const prompt = "Det var";

// sampling-vegen: conf finst, og er ein gyldig sannsyn per valt teikn
const os = { temperature: 0.9, topK: 5, length: 12 };
const st = sampleTokens(m, tok.encode, prompt, os, mulberry32(9));
assert.equal(st.conf.length, st.contIds.length, "conf must be one value per generated token");
assert.equal(st.conf.length, 12);
for (const p of st.conf) {
  assert.ok(p > 0 && p <= 1, `confidence out of range: ${p}`);
}

// grådig-vegen (temperatur 0) må òg fylle conf – slideren når 0
const og = { temperature: 0, topK: 5, length: 10 };
const gd = sampleTokens(m, tok.encode, prompt, og, mulberry32(1));
assert.equal(gd.conf.length, 10);
for (const p of gd.conf) {
  assert.ok(p > 0 && p <= 1, `greedy confidence out of range: ${p}`);
}

// grådig vel alltid det mest sannsynlege teiknet: conf må vere maksimum,
// altså minst 1/V for eit kvart ordforråd
for (const p of gd.conf) {
  assert.ok(p >= 1 / tok.vocab, "greedy pick must be the argmax of the distribution");
}

// conf er uavhengig av temperatur og top-k: same modell, same frø, same
// valde teikn => same sikkerheit. Grådig med topK 5 og topK 50 er identisk.
const a = sampleTokens(m, tok.encode, prompt, { temperature: 0, topK: 5, length: 8 }, mulberry32(3));
const b = sampleTokens(m, tok.encode, prompt, { temperature: 0, topK: 50, length: 8 }, mulberry32(3));
assert.deepEqual(Array.from(a.contIds), Array.from(b.contIds));
for (let i = 0; i < a.conf.length; i++) {
  assert.ok(Math.abs(a.conf[i] - b.conf[i]) < 1e-6, "conf must not depend on top-k");
}

// generateDetailed er nøyaktig generate, pluss tala
const d = generateDetailed(m, tok.decode, tok.encode, prompt, og, mulberry32(1));
const g = generate(m, tok.decode, tok.encode, prompt, og, mulberry32(1));
assert.equal(d.text, g);
assert.equal(d.promptLen, prompt.length);
assert.equal(d.conf.length, 10);
// teikn-nivå tokenisering: framhaldet har eitt teikn per conf-verdi
assert.equal(d.text.length - d.promptLen, d.conf.length);

console.log("conf: PASS");
