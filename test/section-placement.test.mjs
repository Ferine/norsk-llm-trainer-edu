import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
const trainingStart = app.indexOf('id="trening"');
const inspectStart = app.indexOf('id="inspect"');
const chatStart = app.indexOf('id="chat"');
const rlhfStart = app.indexOf('id="rlhf"');

assert.ok(trainingStart >= 0 && inspectStart > trainingStart, "steps 5 and 6 must be present in order");
assert.ok(chatStart > inspectStart && rlhfStart > chatStart, "steps 7 and 8 must be present in order");

const step5 = app.slice(trainingStart, inspectStart);
const step7 = app.slice(chatStart, rlhfStart);
const step8 = app.slice(rlhfStart, app.indexOf('id="eigentekst"'));
assert.doesNotMatch(step5, /<ContextWindow\b/, "the context window must not appear in step 5");
assert.match(step7, /<ContextWindow\b/, "the context window must appear in step 7");
assert.equal((app.match(/<ContextWindow\b/g) ?? []).length, 1, "render the context window exactly once");
assert.match(step8, /<InstructTraining\b/, "step 8 must explain instruction training");
assert.ok(
  step8.indexOf("<InstructTraining") < step8.indexOf("<Rlhf"),
  "instruction training must be explained before the preference-training exercise"
);

console.log("section placement: context window is in step 7 and the SFT bridge starts step 8");
