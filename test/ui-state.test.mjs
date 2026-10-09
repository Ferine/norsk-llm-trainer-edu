import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createElement, useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { Transformer, mulberry32 } from "./dist/ml.js";
import { buildTokenizer } from "./dist/corpus.js";

// Use the existing Vite transformer and React server renderer; no browser or
// extra test dependencies are needed for these state/render regressions.
const server = await createServer({
  configFile: false,
  resolve: {alias: {"@": fileURLToPath(new URL("../src", import.meta.url))}},
  server: {middlewareMode: true, ws: false},
  appType: "custom",
});
const originalWindow = globalThis.window;
try {
  const {default: LossChart} = await server.ssrLoadModule("/src/components/LossChart.tsx");
  const {default: Inspector} = await server.ssrLoadModule("/src/components/Inspector.tsx");
  const {useRlhf} = await server.ssrLoadModule("/src/lib/useRlhf.ts");
  const {STRINGS} = await server.ssrLoadModule("/src/lib/i18n.ts");

  for (const lang of ["bm", "nn"]) {
    const heldout = [{step: 0, loss: 4}, {step: 50, loss: 2.5}, {step: 100, loss: 2.8}];
    const renderChart = (before, after, points = heldout) => renderToStaticMarkup(createElement(LossChart, {
      data: [...Array(50).fill(before), ...Array(50).fill(after)],
      heldout: points, vocab: 50, loss: STRINGS[lang].loss,
    }));
    assert.ok(!renderChart(2, 2.5).includes('role="status"'), `${lang}: both losses rising is not overfitting`);
    assert.ok(!renderChart(2, 2).includes('role="status"'), `${lang}: flat training loss is not an improvement`);
    assert.ok(renderChart(2.5, 2).includes('role="status"'), `${lang}: training improves while held-out worsens`);
    assert.ok(!renderChart(2.5, 2, heldout.slice(0, 2)).includes('role="status"'), `${lang}: no held-out deterioration`);
    assert.ok(!renderChart(2.5, 2, []).includes('role="status"'), `${lang}: optional held-out curve`);
    assert.ok(!renderChart(2.5, 2, [...heldout.slice(0, 2), {step: 100, loss: 2.6}]).includes('role="status"'), `${lang}: ignore small test fluctuations`);
  }

  const tokenizer = buildTokenizer("abc");
  const model = new Transformer({vocab: 3, dim: 8, nLayer: 1, nHead: 2, seqLen: 8, ffnMult: 2}, mulberry32(1));
  const initialWeights = model.params.map(p => p.d.slice());
  const timers = new Map();
  let timerId = 0;
  globalThis.window = {
    setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: (id) => timers.delete(id),
  };
  function tick() {
    const [id, fn] = timers.entries().next().value;
    timers.delete(id);
    fn();
  }
  function drainTimers() {
    let ticks = 0;
    while (timers.size) {
      assert.ok(++ticks < 10_000, "scheduled tuning must finish");
      tick();
    }
  }

  const inspect = model.inspect.bind(model);
  let inspectCalls = 0;
  let countInspect = true;
  model.inspect = (...args) => {
    if (countInspect) inspectCalls++;
    return inspect(...args);
  };
  const args = {getModel: () => model, getTokenizer: () => tokenizer, isTrained: () => true, baseRunning: false};
  const revisions = [0, 0, 0, 0, 5, 5, 65, 65, 66, 66];
  const expectedCalls = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4];

  function Harness() {
    const [phase, setPhase] = useState(0);
    const rlhf = useRlhf(args);
    const props = {model, tokenizer, corpusIds: null, step: 100, modelRevision: rlhf.modelRevision, defaultText: "abc", s: STRINGS.bm.inspect};
    // Calling the component here keeps its hooks on the harness's persistent
    // render-phase rerenders. Separate server renders would reset useMemo and
    // conceal the stale-inspector bug this test needs to catch.
    const view = Inspector(props);
    assert.equal(rlhf.modelRevision, revisions[phase], `revision at phase ${phase}`);
    assert.equal(inspectCalls, expectedCalls[phase], `memo invalidation at phase ${phase}`);
    countInspect = false;
    const fresh = renderToStaticMarkup(createElement(Inspector, props));
    countInspect = true;
    assert.equal(renderToStaticMarkup(view), fresh, `inspector matches current weights at phase ${phase}`);

    if (phase === 0) rlhf.start();
    if (phase === 1) tick(); // generate the first pair
    if (phase === 2) rlhf.choose("A"); // schedule live tuning
    if (phase === 3) {
      assert.ok(rlhf.dpoRunning, "busy state paints before tuning starts");
      drainTimers(); // five full steps, then generate the next pair
    }
    if (phase === 4) rlhf.trainMore();
    if (phase === 5) drainTimers(); // sixty additional steps
    if (phase === 6) rlhf.trainMore();
    if (phase === 7) rlhf.resetTuning(); // rollback and cancel the pending loop
    if (phase === 8) {
      model.params.forEach((p, i) => assert.deepEqual(p.d, initialWeights[i], "rollback restores weights"));
      rlhf.reset();
    }
    if (phase < 9) setPhase(phase + 1);
    else assert.equal(timers.size, 0, "reset clears outstanding work");
    return view;
  }
  renderToStaticMarkup(createElement(Harness));
  console.log("UI state: overfitting trends and inspector refresh after tuning, chunks and rollback ok");
} finally {
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
  await server.close();
}
