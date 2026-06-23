import assert from "node:assert/strict";
import { STRINGS, SEEDS, LANGS } from "./dist/i18n.js";

// recursive key-shape comparison; functions and strings are leaves
function shape(v) {
  if (Array.isArray(v)) return v.map(shape);
  if (v && typeof v === "object") {
    const o = {};
    for (const k of Object.keys(v).sort()) o[k] = shape(v[k]);
    return o;
  }
  return typeof v; // "string" | "function" | "number"
}

assert.deepEqual(shape(STRINGS.bm), shape(STRINGS.nn), "STRINGS bm/nn key shapes must match");
assert.deepEqual(shape(SEEDS.bm), shape(SEEDS.nn), "SEEDS bm/nn key shapes must match");
assert.equal(LANGS.length, 2);
assert.equal(LANGS[0].id, "bm", "Bokmål must be first (default)");
assert.deepEqual(SEEDS.bm.examples.length, SEEDS.nn.examples.length, "same number of example seeds");

console.log("i18n-parity: OK");
