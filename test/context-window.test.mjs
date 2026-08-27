import assert from "node:assert/strict";
import { sliceContextWindow } from "./dist/context-window.js";

const short = sliceContextWindow(["æ", " ", "ø"], 5);
assert.deepEqual(short.visible, ["æ", " ", "ø"]);
assert.equal(short.used, 3);
assert.equal(short.forgotten, 0);
assert.deepEqual(short.forgottenPreview, []);

const sliding = sliceContextWindow(Array.from("abcdefghijkl"), 5, 3);
assert.deepEqual(sliding.visible, Array.from("hijkl"));
assert.equal(sliding.used, 5);
assert.equal(sliding.forgotten, 7);
assert.deepEqual(sliding.forgottenPreview, Array.from("efg"));

assert.throws(() => sliceContextWindow([], 0), /positive integer/);
assert.throws(() => sliceContextWindow([], 4, -1), /non-negative integer/);

console.log("context window: sliding character-token view ok");
