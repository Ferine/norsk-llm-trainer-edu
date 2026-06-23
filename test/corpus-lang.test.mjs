import assert from "node:assert/strict";
import { corpus, corpora } from "./dist/corpus.js";

assert.ok(corpora.bm.length > 200, "bokmål corpus should be non-trivial");
assert.ok(corpora.nn.length > 200, "nynorsk corpus should be non-trivial");
assert.notEqual(corpora.bm, corpora.nn, "the two corpora must differ");
assert.equal(corpus, corpora.bm, "default corpus export must be bokmål");
// Bokmål markers absent from Nynorsk: "ikke" / "hvor" should be in bm, "ikkje"/"kor" in nn
assert.ok(corpora.bm.includes("ikke"), "bokmål uses 'ikke'");
assert.ok(corpora.nn.includes("ikkje"), "nynorsk uses 'ikkje'");

console.log("corpus-lang: OK");
