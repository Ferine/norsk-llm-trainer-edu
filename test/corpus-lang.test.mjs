import assert from "node:assert/strict";
import { corpus, corpora, proveTekster } from "./dist/corpus.js";

assert.ok(corpora.bm.length > 200, "bokmål corpus should be non-trivial");
assert.ok(corpora.nn.length > 200, "nynorsk corpus should be non-trivial");
assert.notEqual(corpora.bm, corpora.nn, "the two corpora must differ");
assert.equal(corpus, corpora.bm, "default corpus export must be bokmål");
// Bokmål markers absent from Nynorsk: "ikke" / "hvor" should be in bm, "ikkje"/"kor" in nn
assert.ok(corpora.bm.includes("ikke"), "bokmål uses 'ikke'");
assert.ok(corpora.nn.includes("ikkje"), "nynorsk uses 'ikkje'");

// Prøveteksten (held-out) må kunne tokeniserast utan tap, og må vere ny tekst.
for (const lang of ["bm", "nn"]) {
  const train = new Set(corpora[lang]);
  const prove = proveTekster[lang];
  assert.ok(prove.length > 300, `${lang}: held-out text should be non-trivial`);
  const missing = [...new Set(prove)].filter((c) => !train.has(c));
  assert.deepEqual(missing, [], `${lang}: held-out uses characters absent from the corpus`);
  for (const sentence of prove.split(/(?<=[.?])\s+/)) {
    assert.ok(!corpora[lang].includes(sentence), `${lang}: held-out sentence is in the corpus: ${sentence}`);
  }
}

console.log("corpus-lang: OK");
