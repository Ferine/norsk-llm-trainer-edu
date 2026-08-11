import assert from "node:assert/strict";
import { ORDLISTE, glossify } from "./dist/ordliste.js";

// Kvar id skal ha begge målformer, med minst éin skrivemåte og ei forklaring
// som faktisk forklarer noko.
for (const [id, entry] of Object.entries(ORDLISTE)) {
  for (const lang of ["bm", "nn"]) {
    const f = entry[lang];
    assert.ok(f, `${id}: manglar ${lang}`);
    assert.ok(f.ord.length >= 1, `${id}/${lang}: ingen skrivemåtar`);
    assert.ok(f.def.length > 30, `${id}/${lang}: forklaringa er for kort til å hjelpe nokon`);
    for (const o of f.ord) {
      assert.equal(o, o.trim(), `${id}/${lang}: skrivemåte med luft rundt: «${o}»`);
    }
  }
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

// Målform-oppslaget skil bm og nn: «merksemd» finst berre i nn-lista.
assert.equal(glossify("multi-head merksemd", "bm").filter((d) => typeof d !== "string").length, 0);
assert.equal(glossify("multi-head merksemd", "nn").filter((d) => typeof d !== "string").length, 1);

console.log(`ordliste: OK – ${Object.keys(ORDLISTE).length} fagord med gloser på bm+nn`);
