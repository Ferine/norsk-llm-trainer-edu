// Proves the exported workbook is a working language model: evaluate its
// formulas with the tiny engine in formula-eval.mjs and compare against ml.ts.

import assert from "node:assert/strict";
import { Transformer, Adam, mulberry32, trainStep } from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";
import { buildModelWorkbook } from "./dist/excel-model.js";
import { buildXlsxParts } from "./dist/xlsx.js";
import { makeEngine } from "./formula-eval.mjs";

const text = "det var en gang en fisker som rodde ut pa fjorden. han sa en sel.\n";
const tokenizer = buildTokenizer(text);
const cfg = {
  vocab: tokenizer.vocab,
  dim: 8,
  nLayer: 2,
  nHead: 2,
  seqLen: 10,
  ffnMult: 2,
};
const model = new Transformer(cfg, mulberry32(1337));

// Train a little so the weights are not the initial noise — an untrained model
// produces near-tied logits, which is precisely where a layout bug could hide.
const data = tokenizer.encode(text);
const opt = new Adam(model.params, 0.01);
const rng = mulberry32(42);
let loss = 0;
for (let i = 0; i < 40; i++) loss = trainStep(model, opt, data, cfg.seqLen, 2, rng);

const prompt = "det var";
const built = buildModelWorkbook({
  model,
  tokenizer,
  prompt,
  nGen: 4,
  step: 40,
  loss,
  presetName: "test",
  lang: "nn",
});

const T = built.positions;
const P = built.prompt.length;
assert.equal(built.prompt, prompt, "prompt should survive sanitising");
assert.equal(T, Math.min(cfg.seqLen, P + 4 - 1));

const eng = makeEngine(built.workbook);

// ---- 1. token ids per position -------------------------------------------
// Greedy decode in JS, recording what each position's id must be.
const ids = tokenizer.encode(built.prompt);
const expectedNext = new Map(); // position -> argmax id at that position
for (let k = 0; k <= T - P; k++) {
  const logits = model.forward(ids);
  const row = ids.length - 1;
  let best = 0;
  let bestv = -Infinity;
  for (let c = 0; c < cfg.vocab; c++) {
    const v = logits.d[row * cfg.vocab + c];
    if (v > bestv) {
      bestv = v;
      best = c;
    }
  }
  expectedNext.set(row, best);
  if (ids.length < T) ids.push(best);
}
assert.equal(ids.length, T, "greedy decode should fill the sheet");

for (let t = 0; t < T; t++) {
  const got = eng.cell(built.probe.ids[t]);
  assert.equal(got, ids[t], `id at position ${t}`);
}
console.log(`  ids: ${T}/${T} positions match ml.ts`);

// ---- 2. winner ids where the sheet actually predicts ----------------------
for (const [row, want] of expectedNext) {
  const got = eng.cell(built.probe.next[row]);
  assert.equal(got, want, `argmax at position ${row}`);
}
console.log(`  argmax: ${expectedNext.size} predicted positions match ml.ts`);

// ---- 3. the logits themselves, numerically -------------------------------
// float32 in ml.ts vs float64 in the sheet, so compare with a tolerance.
const finalLogits = model.forward(ids);
let worst = 0;
for (let t = 0; t < T; t++) {
  const sheetRow = eng.range(built.probe.logitRows[t]);
  assert.equal(sheetRow.length, cfg.vocab, `logit row ${t} width`);
  for (let v = 0; v < cfg.vocab; v++) {
    const want = finalLogits.d[t * cfg.vocab + v];
    const diff = Math.abs(sheetRow[v] - want);
    if (diff > worst) worst = diff;
  }
}
assert.ok(worst < 1e-3, `logits should track ml.ts (worst |diff| = ${worst})`);
console.log(`  logits: worst |float64 − float32| = ${worst.toExponential(2)}`);

// ---- 4. the finished text -------------------------------------------------
const sheetText = eng.cell(built.probe.output);
assert.equal(sheetText, built.reference, "sheet text vs browser text");
assert.ok(sheetText.startsWith(built.prompt));
assert.equal(
  Array.from(sheetText).length,
  P + (T - P + 1),
  "sheet should emit one character per generated position"
);
console.log(`  text: ${JSON.stringify(sheetText)}`);

// ---- 5. no cycles, and every formula is reachable and evaluable -----------
// makeEngine throws on a cycle, so walking every formula also proves the
// workbook is acyclic.
let formulas = 0;
for (const sh of built.workbook.sheets)
  for (const [r, line] of sh.cells)
    for (const [c, cell] of line)
      if (cell.f !== undefined) {
        formulas++;
        const v = eng.cell(`${sh.name}!$${colLetters(c)}$${r}`);
        assert.ok(
          v !== undefined && (typeof v !== "number" || Number.isFinite(v)),
          `${sh.name}!${colLetters(c)}${r} = ${v} (${cell.f})`
        );
      }
assert.equal(formulas, built.formulaCells);
console.log(`  evaluated all ${formulas} formulas, no cycles, no errors`);

function colLetters(col) {
  let n = col;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = (n - 1 - rem) / 26;
  }
  return out;
}

// ---- 6. the .xlsx parts are present and well-formed enough ----------------
const parts = buildXlsxParts(built.workbook);
const paths = parts.map((p) => p.path);
for (const want of [
  "[Content_Types].xml",
  "_rels/.rels",
  "xl/workbook.xml",
  "xl/_rels/workbook.xml.rels",
  "xl/styles.xml",
])
  assert.ok(paths.includes(want), `missing part ${want}`);
assert.equal(
  paths.filter((p) => p.startsWith("xl/worksheets/")).length,
  built.workbook.sheets.length
);

for (const part of parts) {
  // Bare ampersands and angle brackets are the classic way to produce a file
  // Excel refuses to open, so check the escaping held.
  const stripped = part.text.replace(/&(amp|lt|gt|quot|apos|#\d+);/g, "");
  assert.ok(!stripped.includes("&"), `unescaped & in ${part.path}`);
  const opens = (part.text.match(/<[A-Za-z?]/g) ?? []).length;
  const closes = (part.text.match(/(?:\/>|<\/)/g) ?? []).length;
  assert.ok(opens > 0 && closes > 0, `${part.path} looks empty`);
}
assert.ok(parts.some((p) => p.text.includes('fullCalcOnLoad="1"')), "needs recalc-on-open");
const wbXml = parts.find((p) => p.path === "xl/workbook.xml").text;
for (const n of ["Ledetekst", "Teikn", "Kodepunkt", "tokEmb", "posEmb"])
  assert.ok(wbXml.includes(`name="${n}"`), `missing defined name ${n}`);

// ---- 7. both languages produce the same machine, different words ----------
const bm = buildModelWorkbook({
  model, tokenizer, prompt, nGen: 4, step: 40, loss, presetName: "test", lang: "bm",
});
assert.equal(bm.formulaCells, built.formulaCells, "language must not change the maths");
assert.equal(bm.valueCells, built.valueCells);
assert.equal(bm.reference, built.reference);
assert.deepEqual(
  bm.workbook.sheets.map((s) => s.name),
  built.workbook.sheets.map((s) => s.name)
);
assert.equal(makeEngine(bm.workbook).cell(bm.probe.output), bm.reference);
// The label text really is different, and bokmål does not leak nynorsk terms.
const labelsOf = (b) =>
  b.workbook.sheets
    .flatMap((s) => [...s.cells.values()].flatMap((l) => [...l.values()]))
    .map((c) => c.s)
    .filter(Boolean)
    .join("\n");
const bmText = labelsOf(bm);
const nnText = labelsOf(built);
assert.notEqual(bmText, nnText, "the two languages should read differently");
for (const nynorskism of ["merksemd", "Hovud ", "teikn modellen", "kolonnar"])
  assert.ok(!bmText.includes(nynorskism), `bokmål leaks "${nynorskism}"`);
for (const bokmalism of ["oppmerksomhet", "Hode ", "tegn modellen", "kolonner"])
  assert.ok(!nnText.includes(bokmalism), `nynorsk leaks "${bokmalism}"`);
console.log("  language: bm and nn agree on the maths and differ on the words");

const bytes = parts.reduce((n, p) => n + p.text.length, 0);
console.log(
  `  workbook: ${built.workbook.sheets.length} sheets, ${built.formulaCells} formulas, ` +
    `${built.valueCells} values, ${(bytes / 1024).toFixed(0)} KiB of XML`
);
console.log("excel: PASS");
