// Proves the exported workbook is a working language model: evaluate its
// formulas with the tiny engine in formula-eval.mjs and compare against ml.ts.
// Run for both activations — the sheet has to follow whatever the model does.

import assert from "node:assert/strict";
import { Transformer, Adam, mulberry32, trainStep } from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";
import { buildModelWorkbook } from "./dist/excel-model.js";
import { buildXlsxParts, STYLE_INPUT } from "./dist/xlsx.js";
import { makeEngine } from "./formula-eval.mjs";

const text = "det var en gang en fisker som rodde ut pa fjorden. han sa en sel.\n";
const tokenizer = buildTokenizer(text);

for (const act of ["gelu", "situ"]) check(act);

function check(act) {
  console.log(`activation ${act}:`);
  const cfg = {
    vocab: tokenizer.vocab,
    dim: 8,
    nLayer: 2,
    nHead: 2,
    seqLen: 10,
    ffnMult: 2,
    act,
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

  // ---- 6. exactly one cell in the file is editable --------------------------
  // The Les_meg sheet promises "nøyaktig én slik celle i hele filen". That is a
  // claim about the workbook, so assert it rather than trust it.
  const inputCells = [];
  for (const sh of built.workbook.sheets)
    for (const [row, line] of sh.cells)
      for (const [col, cell] of line)
        if (cell.st === STYLE_INPUT) inputCells.push(`${sh.name}!$${colLetters(col)}$${row}`);
  assert.equal(inputCells.length, 1, `expected one input cell, got ${inputCells.join(", ")}`);
  const ledetekst = built.workbook.definedNames.find((d) => d.name === "Ledetekst");
  assert.equal(inputCells[0], ledetekst.ref, "the yellow cell must be the one the model reads");
  assert.equal(eng.cell(inputCells[0]), built.prompt);

  // ---- 7. the flowchart sheet ----------------------------------------------
  const flow = built.workbook.sheets[1];
  assert.equal(flow.name, "Flytskjema", "the flowchart should be the second tab");

  // Every numbered step needs all four description columns filled, in both
  // languages — a missing key would otherwise ship as a silent blank box.
  let steps = 0;
  const liveValues = [];
  for (const [row, line] of flow.cells) {
    const first = line.get(1);
    if (!first || typeof first.n !== "number") continue;
    steps++;
    for (const col of [2, 3, 4, 5]) {
      const cell = line.get(col);
      assert.ok(cell, `flow row ${row} is missing column ${colLetters(col)}`);
      const text = cell.f !== undefined ? eng.cell(`Flytskjema!$${colLetters(col)}$${row}`) : cell.s;
      assert.ok(
        text !== undefined && String(text).length > 0,
        `flow row ${row}, column ${colLetters(col)} is empty`
      );
    }
    const val = line.get(5);
    if (val.f !== undefined) liveValues.push(eng.cell(`Flytskjema!$E$${row}`));
  }
  assert.equal(steps, 18, "the flowchart should have 18 steps");
  assert.ok(liveValues.length >= 15, `expected live values on most steps, got ${liveValues.length}`);

  // The live column is wired to the real model, not to copies of it: the first
  // step must show the prompt, and the character step must show exactly what
  // ml.ts predicts after the last character the user typed.
  assert.ok(liveValues.includes(built.prompt), "step 1 should show the prompt itself");
  const wantChar = tokenizer.itos[expectedNext.get(P - 1)];
  assert.ok(
    liveValues.includes(wantChar),
    `the character step should show ${JSON.stringify(wantChar)}, got ${JSON.stringify(liveValues)}`
  );
  for (const v of liveValues)
    assert.ok(
      typeof v !== "number" || Number.isFinite(v),
      `a live value went non-finite: ${v}`
    );
  console.log(`  flow: ${steps} steps, ${liveValues.length} live values, 1 editable cell`);

  // ---- 8. the .xlsx parts are present and well-formed enough ----------------
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

  // Excel refuses to open a file whose style table declares a count that does
  // not match the number of children — and it says only "unreadable content",
  // so this is worth asserting rather than discovering by hand.
  const styles = parts.find((p) => p.path === "xl/styles.xml").text;
  for (const [tag, child] of [
    ["fonts", "font"],
    ["fills", "fill"],
    ["borders", "border"],
    ["cellXfs", "xf"],
  ]) {
    const declared = Number(styles.match(new RegExp(`<${tag} count="(\\d+)"`))[1]);
    const body = styles.match(new RegExp(`<${tag} count="\\d+">([\\s\\S]*?)</${tag}>`))[1];
    const actual = (body.match(new RegExp(`<${child}[ />]`, "g")) ?? []).length;
    assert.equal(actual, declared, `<${tag}> declares ${declared} but holds ${actual}`);
  }
  // The styles the sheets actually reference must exist in that table.
  const xfCount = Number(styles.match(/<cellXfs count="(\d+)"/)[1]);
  for (const sh of built.workbook.sheets)
    for (const line of sh.cells.values())
      for (const cell of line.values())
        if (cell.st !== undefined)
          assert.ok(cell.st < xfCount, `style ${cell.st} is outside the table of ${xfCount}`);
}

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

console.log("excel: ok");
