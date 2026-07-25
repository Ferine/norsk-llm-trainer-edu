import {
  lossToFocus,
  confToSmudge,
  meanConf,
  blurPx,
  chalkOpacity,
  supportsElementTexture,
  forcedTier,
} from "./dist/chalk.js";
import assert from "node:assert/strict";

const V = 50;
const RANDOM = Math.log(V); // ~3.912 – tapet ved rein gjetting

// nullpunktet er rein gjetting, nøyaktig
assert.ok(Math.abs(lossToFocus(RANDOM, V)) < 1e-9, "ln(V) must map to exactly 0 focus");

// monotont fallande i tap
let prev = -1;
for (const loss of [RANDOM, 3.0, 2.5, 2.0, 1.5, 1.3]) {
  const f = lossToFocus(loss, V);
  assert.ok(f > prev, `focus must increase as loss falls (loss=${loss})`);
  prev = f;
}

// klemt til [0, 1] i begge endar
assert.equal(lossToFocus(99, V), 0, "loss worse than random clamps to 0");
assert.equal(lossToFocus(0.1, V), 1, "loss below the floor clamps to 1");
for (const loss of [0.1, 1.0, 2.0, 4.0, 99]) {
  const f = lossToFocus(loss, V);
  assert.ok(f >= 0 && f <= 1, `focus out of range: ${f}`);
}

// smudge: sikker => skarpt, usikker => uklart, monotont
assert.equal(confToSmudge(1), 0, "full confidence is perfectly crisp");
assert.ok(confToSmudge(0.01) > 0.8, "near-zero confidence is heavily smudged");
assert.ok(confToSmudge(0.9) < confToSmudge(0.3), "smudge must fall as confidence rises");
for (const p of [0, 0.25, 0.5, 0.75, 1]) {
  const s = confToSmudge(p);
  assert.ok(s >= 0 && s <= 1, `smudge out of range: ${s}`);
}
// robust mot søppel-input
assert.equal(confToSmudge(-1), 1);
assert.equal(confToSmudge(2), 0);

// snittsikkerheit
assert.ok(Math.abs(meanConf(new Float32Array([0.5, 0.5, 0.5])) - 0.5) < 1e-6);
assert.ok(Math.abs(meanConf(new Float32Array([0.2, 0.8])) - 0.5) < 1e-6);
assert.equal(meanConf(new Float32Array([])), 0, "empty conf must not be NaN");

// css-avbildingar er endelege og monotone
assert.equal(blurPx(0), 0);
assert.ok(blurPx(1) > blurPx(0.5) && blurPx(0.5) > 0);
assert.equal(chalkOpacity(0), 1);
assert.ok(chalkOpacity(1) < chalkOpacity(0.5) && chalkOpacity(1) > 0);

// nettlesar-deteksjon må ikkje krasje i Node
assert.equal(supportsElementTexture(), false, "no WebGL2 in Node");
assert.equal(forcedTier(), null, "no location in Node");

console.log("chalk: PASS");
