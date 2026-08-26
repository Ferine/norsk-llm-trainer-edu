// A source-level guard: every control that throws away training has to go
// through the confirmation dialog. This is a policy the UI can silently break
// when someone adds a new switch, and there is no React harness here to click
// it, so the policy is asserted against the source instead.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
const rlhfView = readFileSync(new URL("../src/components/Rlhf.tsx", import.meta.url), "utf8");
const i18n = readFileSync(new URL("../src/lib/i18n.ts", import.meta.url), "utf8");

// ---- 1. state setters that force a rebuild are always guarded --------------
// buildEngine's dependency list decides what discards the model. Keep this list
// in step with it: preset, act, moe, ngram and optim rebuild directly, lang rebuilds
// through activeCorpus.
const DESTRUCTIVE = ["setPreset", "setOptim", "setAct", "setMoe", "setNgram", "setLang"];
for (const setter of DESTRUCTIVE) {
  const calls = [...app.matchAll(new RegExp(`\\b${setter}\\(`, "g"))];
  assert.ok(calls.length > 0, `${setter} should be called somewhere`);
  for (const m of calls) {
    const before = app.slice(Math.max(0, m.index - 160), m.index);
    assert.ok(
      /guard\(\s*"[a-z]+"/.test(before),
      `${setter}( at index ${m.index} is not inside a guard(...) call:\n…${before.slice(-120)}`
    );
  }
}

// ---- 2. buildEngine's deps have not grown a new rebuild trigger ------------
// If this fails, something new discards the model and needs a guard + a key in
// DESTRUCTIVE above.
const deps = app.match(/\}, \[preset, act, moe, ngram, optim, rlhf\.reset, activeCorpus\]\);/);
assert.ok(
  deps,
  "buildEngine's dependency list changed — check whether the new dependency discards training, and guard it"
);

// ---- 3. the destructive buttons are guarded --------------------------------
assert.ok(
  /const onResetClick = useCallback\(\(\) => guard\("reset", reset\)/.test(app),
  "the reset button must ask first"
);
assert.ok(
  /const rebuildWithExtraText = useCallback\(\s*\(\) =>\s*guard\("text"/.test(app),
  "rebuilding with your own text must ask first"
);
assert.ok(
  /guard\("restart", beginTraining\)/.test(app),
  "starting a fresh run over a finished one must ask first"
);

// Continuing a paused run must NOT ask — only the rebuild path does.
const startBody = app.slice(app.indexOf("const start = useCallback"), app.indexOf("startFromHero"));
assert.ok(
  /stepRef\.current >= MAX_STEPS/.test(startBody),
  "start should only ask when the previous run is finished"
);
assert.ok(startBody.includes("beginTraining();"), "start should continue a paused run directly");

// ---- 4. the RLHF rollback goes through App, which asks ---------------------
assert.ok(
  !/onClick=\{rlhf\.resetTuning\}/.test(rlhfView),
  "resetTuning rolls weights back to the reference — route it through onResetTuning"
);
assert.ok(rlhfView.includes("onClick={onResetTuning}"), "the tuning reset button should be guarded");
assert.ok(
  /rlhf\.metrics\.count === 0/.test(app),
  "asking about tuning should be skipped when there is no tuning to lose"
);

// ---- 5. no confirmation is shown when there is nothing to lose ------------
assert.ok(
  /if \(stepRef\.current === 0\) \{\s*run\(\);/.test(app),
  "an untrained model should be swapped without a dialog"
);
// Clicking the language you are already in changes nothing, so it must not ask.
assert.ok(
  /if \(l\.id !== lang\) guard\("lang"/.test(app),
  "picking the current language should not open a dialog"
);

// ---- 6. the copy exists in both languages ---------------------------------
// i18n-parity checks the shape; this checks the keys are actually the ones the
// dialog reads, so a rename cannot leave the dialog blank.
for (const key of ["preset", "optim", "act", "moe", "ngram", "lang", "text", "restart", "reset"]) {
  const uses = [...i18n.matchAll(new RegExp(`^\\s{6}${key}: "`, "gm"))];
  assert.equal(uses.length, 2, `confirm.what.${key} should exist in both bm and nn`);
}
for (const key of ["title", "body", "yes", "no"]) {
  assert.ok(
    new RegExp(`^\\s{4}${key}:`, "m").test(i18n),
    `confirm.${key} is missing from the strings`
  );
}

console.log("confirm: ok");
