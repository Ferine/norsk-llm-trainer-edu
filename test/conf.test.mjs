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
// promptLen er eit kodepunkt-tal, ikkje eit UTF-16-lengde-tal (dei er like
// for denne promten sidan han berre har BMP-teikn, men samanlikninga skal
// framleis vere mot kodepunkt-talet, ikkje mot .length)
assert.equal(d.promptLen, Array.from(prompt).length);
assert.equal(d.conf.length, 10);
// teikn-nivå tokenisering: framhaldet har eitt teikn per conf-verdi
assert.equal(Array.from(d.text).length - d.promptLen, d.conf.length);

// --- Fix pass (code review, finding 1): promptLen må telje kodepunkt,
// ikkje UTF-16-einingar, elles hamnar heile smugekartet i Tavle.tsx eitt
// hakk feil for ein prompt med eit astralt teikn (t.d. emoji) ---
{
  const emojiPrompt = "Det var 🐑 en gang";
  // 🐑 er eitt kodepunkt men to UTF-16-einingar – nettopp skilnaden
  // reproen målte (promptLen=18 vs. 17 kodepunkt)
  assert.equal(emojiPrompt.length, 18, "sanity: UTF-16 length counts the emoji as two units");
  assert.equal(Array.from(emojiPrompt).length, 17, "sanity: code-point length counts the emoji as one");

  const opts2 = { temperature: 0.9, topK: 5, length: 20 };
  const ed = generateDetailed(m, tok.decode, tok.encode, emojiPrompt, opts2, mulberry32(11));
  const st2 = sampleTokens(m, tok.encode, emojiPrompt, opts2, mulberry32(11));

  assert.equal(ed.promptLen, Array.from(emojiPrompt).length, "promptLen must be the code-point count");

  // Sjølve eigenskapen som var øydelagd: for kvar genererte posisjon i skal
  // kodepunktet i teksten på plass promptLen+i vere nøyaktig det teiknet
  // conf[i]/contIds[i] gjeld for – ikkje eit teikn frå naboposisjonen.
  const codePoints = Array.from(ed.text);
  for (let i = 0; i < st2.contIds.length; i++) {
    assert.equal(
      codePoints[ed.promptLen + i],
      tok.decode([st2.contIds[i]]),
      `generated char at position ${i} must align with its own token, not a shifted neighbour`
    );
  }
}

console.log("conf: PASS");
