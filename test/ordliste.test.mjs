import assert from "node:assert/strict";
import { ORDLISTE, TEMA_REKKEFOLGJE, glossify, ordlisteTema } from "./dist/ordliste.js";

// Kvar id skal ha eit tema og begge målformer, med oppslagsform, minst éin
// skrivemåte og ei forklaring som faktisk forklarer noko.
for (const [id, entry] of Object.entries(ORDLISTE)) {
  assert.ok(TEMA_REKKEFOLGJE.includes(entry.tema), `${id}: ukjent tema «${entry.tema}»`);
  for (const lang of ["bm", "nn"]) {
    const f = entry[lang];
    assert.ok(f, `${id}: manglar ${lang}`);
    assert.ok(f.vis.length >= 2, `${id}/${lang}: manglar oppslagsform`);
    assert.ok(f.ord.length >= 1, `${id}/${lang}: ingen skrivemåtar`);
    assert.ok(f.def.length > 30, `${id}/${lang}: forklaringa er for kort til å hjelpe nokon`);
    for (const o of f.ord) {
      assert.equal(o, o.trim(), `${id}/${lang}: skrivemåte med luft rundt: «${o}»`);
    }
  }
}

// Ordlista (steg 11) skal dekkje kvart oppslag nøyaktig éin gong, i temarekkjefølgje.
for (const lang of ["bm", "nn"]) {
  const grupper = ordlisteTema(lang);
  assert.deepEqual(
    grupper.map((g) => g.tema),
    TEMA_REKKEFOLGJE,
    `${lang}: gruppene skal følgje temarekkjefølgja`
  );
  const ids = grupper.flatMap((g) => g.oppslag.map((o) => o.id));
  assert.deepEqual(
    [...ids].sort(),
    Object.keys(ORDLISTE).sort(),
    `${lang}: ordlista skal dekkje alle oppslaga, utan duplikat`
  );
}

// glossify: finn fagord, tek vare på original stor/liten bokstav …
let deler = glossify("Vi bruker en transformer i nettleseren.", "bm");
let treff = deler.filter((d) => typeof d !== "string");
assert.equal(treff.length, 1);
assert.equal(treff[0].ord, "transformer");
assert.equal(deler.map((d) => (typeof d === "string" ? d : d.ord)).join(""), "Vi bruker en transformer i nettleseren.");

// … merkjer berre første førekomst av kvart ord …
deler = glossify("Token her og token der – token overalt.", "bm");
assert.equal(deler.filter((d) => typeof d !== "string").length, 1);

// … lèt bøygde naboord vere i fred («taper» er eit verb, ikkje eit tap) …
treff = glossify("Her taper modellen litt.", "bm").filter((d) => typeof d !== "string");
assert.equal(treff.length, 0);
treff = glossify("Sjå om tapet søkk.", "nn").filter((d) => typeof d !== "string");
assert.equal(treff.length, 1);
assert.equal(treff[0].id, "tap");

// … føretrekkjer lengste skrivemåte («tap (loss)» som eitt treff) …
treff = glossify("Dette tallet kalles tap (loss).", "bm").filter((d) => typeof d !== "string");
assert.equal(treff.length, 1);
assert.equal(treff[0].ord, "tap (loss)");

// … og treffer gjennom bindestrek-samansetjingar som «GGUF-fila».
treff = glossify("GGUF-fila under tek med ekspertane.", "nn").filter((d) => typeof d !== "string");
assert.equal(treff.length, 1);
assert.equal(treff[0].id, "gguf");

// Målform-oppslaget skil bm og nn: «sjølvmerksemd» finst berre i nn-lista.
assert.equal(glossify("sjølvmerksemd", "bm").filter((d) => typeof d !== "string").length, 0);
assert.equal(glossify("sjølvmerksemd", "nn").filter((d) => typeof d !== "string").length, 1);

// «multi-head merksemd» skal vera EITT treff på heile frasen (attention),
// ikkje «multi-head» pluss rest – lengste skrivemåte vinn.
treff = glossify("multi-head merksemd", "nn").filter((d) => typeof d !== "string");
assert.equal(treff.length, 1);
assert.equal(treff[0].id, "attention");

// Trigramminnet in step 11 should explain both the mechanism and its source,
// while the component terms remain independently discoverable.
for (const lang of ["bm", "nn"]) {
  assert.match(ORDLISTE["ngram-minne"][lang].def, /Qwen3\.8-Flash-Next/);
  assert.match(ORDLISTE["ngram-minne"][lang].def, /§2\.3/);
  assert.match(ORDLISTE.hashfunksjon[lang].def, /FNV-1a/);
  for (const id of ["n-gram", "oppslagstabell", "hashkollisjon", "ablasjon", "heldout", "overtilpassing"])
    assert.ok(ORDLISTE[id][lang].def.length > 80, `${id}/${lang}: needs a useful explanation`);
}
treff = glossify(
  "Trigramminnet hashes nøkkelen, og hashkollisjoner deler en rad.",
  "bm"
).filter((d) => typeof d !== "string");
assert.deepEqual(treff.map((d) => d.id), ["ngram-minne", "hashfunksjon", "hashkollisjon"]);

console.log(`ordliste: OK – ${Object.keys(ORDLISTE).length} fagord med gloser på bm+nn`);
