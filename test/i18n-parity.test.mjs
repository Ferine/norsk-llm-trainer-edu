import assert from "node:assert/strict";
import { STRINGS, SEEDS, LANGS, LESELISTE } from "./dist/i18n.js";

// recursive key-shape comparison; functions and strings are leaves
function shape(v) {
  if (Array.isArray(v)) return v.map(shape);
  if (v && typeof v === "object") {
    const o = {};
    for (const k of Object.keys(v).sort()) o[k] = shape(v[k]);
    return o;
  }
  return typeof v; // "string" | "function" | "number"
}

assert.deepEqual(shape(STRINGS.bm), shape(STRINGS.nn), "STRINGS bm/nn key shapes must match");
assert.deepEqual(shape(SEEDS.bm), shape(SEEDS.nn), "SEEDS bm/nn key shapes must match");
assert.equal(LANGS.length, 2);
assert.equal(LANGS[0].id, "bm", "Bokmål must be first (default)");
assert.deepEqual(SEEDS.bm.examples.length, SEEDS.nn.examples.length, "same number of example seeds");
assert.equal(STRINGS.bm.hero.h1Lang, "norsk");
assert.equal(STRINGS.nn.hero.h1Lang, "norsk");
assert.match(STRINGS.bm.bpe.intro, /^Store språkmodeller/);
assert.match(STRINGS.nn.bpe.intro, /^Store språkmodellar/);

// Leselista (steg 10): tsc fanger manglende oversettelser, men ikke en id som
// bare finnes i strengene, en duplikat-id eller en lenke som ikke går ut på nett.
const lenker = LESELISTE.flatMap((h) => h.items);
const ids = lenker.map((l) => l.id);
assert.equal(new Set(ids).size, ids.length, "reading-list ids must be unique");
for (const lang of ["bm", "nn"]) {
  const rm = STRINGS[lang].readMore;
  assert.deepEqual(
    Object.keys(rm.items).sort(),
    [...ids].sort(),
    `${lang}: readMore.items must cover exactly the LESELISTE ids`
  );
  for (const h of LESELISTE) {
    assert.ok(rm.shelves[h.id], `${lang}: missing shelf label for "${h.id}"`);
  }
  for (const l of lenker) {
    assert.ok(rm.kinds[l.kind], `${lang}: missing kind label for "${l.kind}"`);
    assert.ok(rm.levels[l.level], `${lang}: missing level label for "${l.level}"`);
    assert.ok(rm.items[l.id].length > 20, `${lang}: note for "${l.id}" is too short to help anyone`);
  }
}
for (const l of lenker) {
  assert.match(l.url, /^https:\/\/[^\s"']+$/, `${l.id}: must be a plain https URL`);
}
assert.equal(lenker.filter((l) => l.start).length, 1, "exactly one link is marked «start her»");

// The new architecture is traceable to primary, official sources rather than
// an unexplained name in the UI.
const byId = new Map(lenker.map((l) => [l.id, l]));
assert.equal(
  byId.get("qwen38report")?.url,
  "https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf"
);
assert.equal(
  byId.get("qwen38blog")?.url,
  "https://qwen.ai/blog?id=qwen3.8-flash-next"
);
assert.equal(
  byId.get("qwen38config")?.url,
  "https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json"
);
assert.equal(byId.get("instructgpt")?.url, "https://arxiv.org/abs/2203.02155");
for (const lang of ["bm", "nn"]) {
  assert.match(STRINGS[lang].readMore.items.qwen38report, /§2\.3/);
  assert.match(STRINGS[lang].readMore.items.qwen38config, /ngram_size: 3/);
  assert.doesNotMatch(STRINGS[lang].readMore.intro, /trigram|Qwen/i);
  assert.doesNotMatch(STRINGS[lang].ordliste.intro, /trigram|Qwen/i);
  assert.doesNotMatch(STRINGS[lang].chat.context.title, /token|seq_len|kontekst/i);
  assert.match(STRINGS[lang].chat.context.oneChar, /ett token|eitt token/i);
}

console.log(`i18n-parity: OK – ${lenker.length} lenker i leselista, alle med notat på bm+nn`);
