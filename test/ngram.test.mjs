import assert from "node:assert/strict";
import { buildTokenizer } from "./dist/corpus.js";
import { buildModelWorkbook } from "./dist/excel-model.js";
import {
  Adam,
  Muon,
  NGRAM_BOS,
  Transformer,
  cloneTransformer,
  mulberry32,
  ngramKeyAt,
  ngramSlot,
  ngramSlotsFor,
  trainStep,
} from "./dist/ml.js";

// The memory consumes ordinary character IDs; it never merges or inserts a
// token. Include a supplementary Unicode character so this tests characters,
// not UTF-16 code units.
const text = "Blåbær 🫐 og æøå.\n";
const tok = buildTokenizer(text);
const ids = tok.encode(text);
assert.equal(ids.length, Array.from(text).length, "one character must remain one token");
assert.equal(tok.decode(ids), text, "character-token round-trip must stay exact");

const cfg = {
  vocab: tok.vocab,
  dim: 12,
  nLayer: 2,
  nHead: 2,
  seqLen: 8,
  ffnMult: 4,
  act: "situ",
};
const ngram = { size: 3, slots: 64, layer: 1 };
const base = new Transformer(cfg, mulberry32(91));
const mem = new Transformer({ ...cfg, ngram }, mulberry32(91));

assert.equal(mem.params.length, base.params.length + 1, "memory should add one table tensor");
assert.equal(mem.ngramParamCount(), ngram.slots * cfg.dim);
assert.equal(mem.paramCount() - base.paramCount(), ngram.slots * cfg.dim);
for (let p = 0; p < base.params.length; p++)
  assert.deepEqual(
    Array.from(mem.params[p].d),
    Array.from(base.params[p].d),
    `shared parameter ${p} must start byte-identical`
  );

const short = ids.slice(0, 6);
assert.deepEqual(Array.from(ngramKeyAt(short, 0, 3)), [NGRAM_BOS, NGRAM_BOS, short[0]]);
assert.deepEqual(Array.from(ngramKeyAt(short, 1, 3)), [NGRAM_BOS, short[0], short[1]]);
assert.deepEqual(Array.from(ngramKeyAt(short, 2, 3)), short.slice(0, 3));
const slots = ngramSlotsFor(short, tok.vocab, ngram);
assert.deepEqual(slots, ngramSlotsFor(short, tok.vocab, ngram), "hashing must be deterministic");

const futureChanged = short.slice();
futureChanged[5] = (futureChanged[5] + 1) % tok.vocab;
assert.deepEqual(
  Array.from(ngramSlotsFor(futureChanged, tok.vocab, ngram).slice(0, 5)),
  Array.from(slots.slice(0, 5)),
  "a future character must not change an earlier lookup"
);
assert.throws(() => ngramSlot([tok.vocab], tok.vocab, ngram.slots), /outside the vocabulary/);

const inspected = mem.inspect(short);
assert.ok(inspected.ngram, "configured memory should be inspectable");
assert.equal(inspected.ngram.size, 3);
assert.equal(inspected.ngram.layer, 1);
assert.deepEqual(inspected.ngram.buckets, slots);
assert.equal(inspected.ngram.keys.length, short.length * 3);
assert.equal(base.inspect(short).ngram, null, "today's baseline has no lookup table");

// Sparse lookup semantics: only rows consulted by this step can move.
const before = mem.ngramEmb.d.slice();
const used = new Set(ngramSlotsFor(ids.slice(0, 5), tok.vocab, ngram));
const opt = new Adam(mem.params, 0.001);
trainStep(mem, opt, ids, 5, 1, () => 0);
let changedUsed = false;
for (const bucket of used)
  for (let d = 0; d < cfg.dim; d++) {
    const i = bucket * cfg.dim + d;
    if (mem.ngramEmb.d[i] !== before[i]) changedUsed = true;
  }
assert.ok(changedUsed, "at least one consulted memory row should learn");
const unused = Array.from({ length: ngram.slots }, (_, i) => i).find((i) => !used.has(i));
assert.notEqual(unused, undefined);
assert.deepEqual(
  Array.from(mem.ngramEmb.d.slice(unused * cfg.dim, (unused + 1) * cfg.dim)),
  Array.from(before.slice(unused * cfg.dim, (unused + 1) * cfg.dim)),
  "an unconsulted row must stay untouched"
);

const clone = cloneTransformer(mem);
assert.deepEqual(
  Array.from(clone.forward(short).d),
  Array.from(mem.forward(short).d),
  "a cloned trigram model must produce identical logits"
);

assert.throws(
  () =>
    buildModelWorkbook({
      model: mem,
      tokenizer: tok,
      prompt: "Blå",
      nGen: 2,
      step: 1,
      loss: 1,
      presetName: "test",
      lang: "bm",
      includeQuant: false,
    }),
  /trigramminne/,
  "the spreadsheet must refuse a model it cannot faithfully calculate"
);

// The switch composes with the other advanced architecture and optimizer
// controls exposed beside it in the UI.
const mixed = new Transformer(
  {
    ...cfg,
    ngram,
    moe: { experts: 4, topK: 1, bias: 0.001 },
  },
  mulberry32(121)
);
const groups = mixed.optimGroups();
assert.ok(groups.scalar.includes(mixed.ngramEmb), "Muon must keep lookup tables on Adam");
const covered = [...groups.matrix.map((g) => g.p), ...groups.scalar];
assert.equal(new Set(covered).size, mixed.params.length);
assert.ok(Number.isFinite(trainStep(mixed, new Muon(groups, 0.0008), ids, 5, 1, mulberry32(3))));
const mixedView = mixed.inspect(short);
assert.ok(mixedView.ngram && mixedView.routes.length > 0, "memory and experts should inspect together");

assert.throws(
  () => new Transformer({ ...cfg, ngram: { ...ngram, layer: cfg.nLayer } }, mulberry32(1)),
  /existing zero-based layer/
);
assert.throws(
  () => new Transformer({ ...cfg, ngram: { ...ngram, size: 1 } }, mulberry32(1)),
  /at least 2/
);

console.log("ngram memory: tokenizer, hashing, sparse learning, inspection and cloning ok");
