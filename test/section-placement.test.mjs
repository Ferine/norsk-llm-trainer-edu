import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
const trainingStart = app.indexOf('id="trening"');
const dataStart = app.indexOf('id="data"');
const bpeStart = app.indexOf('id="bpe"');
const inspectStart = app.indexOf('id="inspect"');
const chatStart = app.indexOf('id="chat"');
const rlhfStart = app.indexOf('id="rlhf"');

assert.ok(trainingStart >= 0 && inspectStart > trainingStart, "steps 5 and 6 must be present in order");
assert.ok(chatStart > inspectStart && rlhfStart > chatStart, "steps 7 and 8 must be present in order");
assert.ok(dataStart >= 0 && bpeStart > dataStart, "steps 2 and 3 must be present in order");

const step2 = app.slice(dataStart, bpeStart);
const step5 = app.slice(trainingStart, inspectStart);
const step7 = app.slice(chatStart, rlhfStart);
const step8 = app.slice(rlhfStart, app.indexOf('id="eigentekst"'));
assert.doesNotMatch(step5, /<ContextWindow\b/, "the context window must not appear in step 5");
assert.match(step7, /<ContextWindow\b/, "the context window must appear in step 7");
assert.equal((app.match(/<ContextWindow\b/g) ?? []).length, 1, "render the context window exactly once");
assert.match(step2, /<MediaTokens\b/, "image and audio tokenization must be explained in step 2");
assert.match(step8, /<InstructTraining\b/, "step 8 must explain instruction training");
assert.ok(
  step8.indexOf("<InstructTraining") < step8.indexOf("<Rlhf"),
  "instruction training must be explained before the preference-training exercise"
);

// Step 10 – production: the two exports live there now, not in the footer.
const extraStart = app.indexOf('id="eigentekst"');
const prodStart = app.indexOf('id="produksjon"');
const lesmerStart = app.indexOf('id="lesmer"');
const ordlisteStart = app.indexOf('id="ordliste"');
const footerStart = app.indexOf("<footer");
assert.ok(extraStart < prodStart && prodStart < lesmerStart && lesmerStart < ordlisteStart, "step 10 sits between 9 and 11");
const step10 = app.slice(prodStart, lesmerStart);
const footer = app.slice(footerStart);
for (const handler of ["onExcelClick", "onGgufClick"]) {
  assert.match(step10, new RegExp(`onClick=\\{${handler}\\}`), `${handler} must be wired inside step 10`);
  assert.doesNotMatch(footer, new RegExp(handler), `${handler} must no longer be in the footer`);
}
assert.match(step10, /gguf-visualizer/, "the GGUF viewer link belongs with the GGUF download");
assert.match(app.slice(lesmerStart, ordlisteStart), /step=\{11\}/, "the reading list is step 11");
assert.match(app.slice(ordlisteStart), /step=\{12\}/, "the glossary is step 12");

console.log("section placement: media tokens are in step 2, context is in step 7, SFT starts step 8, and the exports live in step 10");
