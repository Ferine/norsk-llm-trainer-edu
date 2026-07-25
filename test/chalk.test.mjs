import {
  lossToFocus,
  confToSmudge,
  meanConf,
  trailingMean,
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

// --- Fix pass (code review, finding 1): golvet må vere relativt til taket,
// elles klapsar span saman for lite vokabular ---

// nullpunktet skal vere eksakt 0 sjølv ved svært lite vokabular, der det
// gamle faste golvet (LOSS_FLOOR=1,25) ville ha lege over taket
assert.ok(
  Math.abs(lossToFocus(Math.log(2), 2)) < 1e-9,
  "ln(2) must map to exactly 0 focus at vocab=2"
);
assert.ok(
  Math.abs(lossToFocus(Math.log(3), 3)) < 1e-9,
  "ln(3) must map to exactly 0 focus at vocab=3"
);

// monotont fallande i tap held også ved vokabular=2, der golvet no er
// halvparten av taket i staden for det faste (og der uoppnåelege) 1,25
{
  let prev2 = -1;
  const ceil2 = Math.log(2);
  const floor2 = ceil2 * 0.5;
  for (const loss of [ceil2, ceil2 * 0.85, ceil2 * 0.7, ceil2 * 0.55, floor2]) {
    const f = lossToFocus(loss, 2);
    assert.ok(f > prev2, `focus must increase as loss falls at vocab=2 (loss=${loss})`);
    prev2 = f;
  }
}

// regresjonsvern: ved realistisk (teiknnivå) vokabular=50 skal golvet
// framleis vere det faste LOSS_FLOOR=1,25 (sidan ceil*0,5 ~1,96 > 1,25), slik
// at biletet på tavla ikkje endrar seg. Rekna direkte mot den gamle
// golv-spannen, slik at ei seinare endring i golvvalet ved realistiske
// vokabularstorleikar feilar høglydt.
{
  const oldSpanAt50 = RANDOM - 1.25;
  const expected = (RANDOM - 2) / oldSpanAt50;
  assert.ok(
    Math.abs(lossToFocus(2, V) - expected) < 1e-9,
    "vocab=50 behaviour must be unchanged from the old fixed LOSS_FLOOR span"
  );
}

// --- Fix pass (code review, finding 2): ikkje-endeleg vokabular må ikkje
// smitte NaN vidare ---
for (const badVocab of [NaN, Infinity, -Infinity]) {
  const f = lossToFocus(1.5, badVocab);
  assert.ok(Number.isFinite(f), `non-finite vocab (${badVocab}) must not propagate to NaN`);
  assert.ok(f >= 0 && f <= 1, `non-finite vocab (${badVocab}) must clamp to [0,1]`);
}

// --- Fix pass (code review, finding 3): manglande grensetest for smudge ---
assert.equal(confToSmudge(0), 1, "fully uncertain is fully smudged");

// --- Final review fix pass (finding 2): trailingMean jamnar ut §5 sin
// målar mot minibatch-støy ---

// tomt array => 0, aldri NaN
assert.equal(trailingMean([], 20), 0, "empty losses must not be NaN");

// vindauge lengre enn tilgjengeleg data => snitt av alt vi har
assert.ok(
  Math.abs(trailingMean([1, 2, 3], 20) - 2) < 1e-9,
  "window wider than the data must average over what's available"
);

// vanleg tilfelle: berre dei siste `window` verdiane skal telje med
assert.ok(
  Math.abs(trailingMean([100, 100, 1, 2, 3], 3) - 2) < 1e-9,
  "must average only the trailing window, ignoring older values"
);

// vindauge = 1 er identisk med siste verdi
assert.equal(trailingMean([5, 6, 7], 1), 7, "window of 1 must equal the last value");

// jamnar faktisk ut støy: eit hakk (ein enkelt outlier) skal dempast mykje
// meir i det glidande snittet enn i den rå siste verdien
{
  const noisy = [55, 53, 59, 56, 57, 55, 56, 58, 56, 90]; // siste er ein outlier
  const last = noisy[noisy.length - 1];
  const smoothed = trailingMean(noisy, 20);
  assert.ok(
    Math.abs(smoothed - last) > 10,
    "a single spike must be damped far more in the trailing mean than in the raw last value"
  );
}

console.log("chalk: PASS");
