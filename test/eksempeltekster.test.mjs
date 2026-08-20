import assert from "node:assert/strict";
import { EKSEMPELTEKSTER } from "./dist/eksempeltekster.js";

// Utdraga skal vera ferdig kuraterte: unike, kronologiske, med kjelde – og
// sjølve teksten skal vera rein flytande prosa utan rusk frå kjeldeformatet.
assert.ok(EKSEMPELTEKSTER.length >= 7, "for få tekster");
const ids = EKSEMPELTEKSTER.map((e) => e.id);
assert.equal(new Set(ids).size, ids.length, "duplikat-id");
for (let i = 1; i < EKSEMPELTEKSTER.length; i++) {
  assert.ok(
    EKSEMPELTEKSTER[i].aar >= EKSEMPELTEKSTER[i - 1].aar,
    "lista skal vera kronologisk"
  );
}

for (const e of EKSEMPELTEKSTER) {
  assert.ok(e.forfattar.length > 3 && e.tittel.length > 2, `${e.id}: manglar opphav`);
  assert.ok(
    e.aar > 1800 && e.aar <= new Date().getFullYear(),
    `${e.id}: mistenkjeleg årstal`
  );
  assert.ok(["fri", "ccbysa"].includes(e.lisens), `${e.id}: ukjend lisens`);
  assert.ok(e.url.startsWith("https://"), `${e.id}: kjelde-URL manglar`);
  assert.ok(e.kjelde.length > 3, `${e.id}: kjeldenamn manglar`);

  const t = e.tekst;
  assert.equal(t, t.trim(), `${e.id}: luft rundt teksten`);
  assert.ok(t.length > 1500 && t.length < 4500, `${e.id}: ${t.length} teikn – utanfor målet`);
  assert.ok(t.includes("\n\n"), `${e.id}: ingen avsnitt`);
  assert.ok(!/[^\n]\n[^\n]/.test(t), `${e.id}: harde linjeskift inni avsnitt`);
  assert.ok(
    !/<|\[\d+\]|Gutenberg|Wikisource/i.test(t),
    `${e.id}: rusk frå kjeldeformatet i teksten`
  );
  assert.ok(!/ -- |“|”|„/.test(t), `${e.id}: unormaliserte teikn`);
}

// Begge målformene skal vera representerte i det moderne laget òg.
assert.ok(EKSEMPELTEKSTER.some((e) => e.lisens === "ccbysa" && e.kjelde.startsWith("no.")));
assert.ok(EKSEMPELTEKSTER.some((e) => e.lisens === "ccbysa" && e.kjelde.startsWith("nn.")));

const totalt = EKSEMPELTEKSTER.reduce((n, e) => n + e.tekst.length, 0);
console.log(`eksempeltekster: OK – ${EKSEMPELTEKSTER.length} tekster, ${totalt} teikn`);
