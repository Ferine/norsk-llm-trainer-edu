import { Fragment, useMemo, useState } from "react";
import { rowProbs, type Transformer } from "@/lib/ml";
import type { Tokenizer } from "@/lib/corpus";
import type { Strings } from "@/lib/i18n";

interface Props {
  model: Transformer | null;
  tokenizer: Tokenizer | null;
  step: number;
  defaultText: string;
  s: Strings["inspect"];
}

const MAX_BARS = 12;

function charLabel(itos: string[], id: number): string {
  const c = itos[id] ?? "";
  if (c === " ") return "␣";
  if (c === "\n") return "⏎";
  return c;
}

export default function Inspector({ model, tokenizer, step, defaultText, s }: Props) {
  const [text, setText] = useState(defaultText);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [pos, setPos] = useState<number | null>(null);

  // One forward pass. Recomputes only when the text, the model instance, or the
  // training step changes (step is an intentional dependency so the panels refresh
  // after a training run). Changing layer/head/pos is a pure re-render off this memo.
  const result = useMemo(() => {
    if (!model || !tokenizer) return null;
    let ids = tokenizer.encode(text);
    if (ids.length === 0) ids = [0];
    if (ids.length > model.seqLen) ids = ids.slice(ids.length - model.seqLen);
    const { logits, attn } = model.inspect(ids);
    return { ids, logits, attn };
  }, [text, model, step, tokenizer]);

  if (!model || !tokenizer || !result) {
    return <p className="text-sm text-slate-500">{s.notReady}</p>;
  }

  const itos = tokenizer.itos;
  const T = result.ids.length;
  const sel = Math.min(pos ?? T - 1, T - 1);
  const nLayer = model.cfg.nLayer;
  const nHead = model.cfg.nHead;
  const layerSel = Math.min(layer, nLayer - 1);
  const headSel = Math.min(head, nHead - 1);

  const view =
    result.attn.find((v) => v.layer === layerSel && v.head === headSel) ?? result.attn[0];

  const probs = rowProbs(result.logits, sel);
  const ranking = Array.from(probs, (p, id) => ({ id, p })).sort((a, b) => b.p - a.p);
  const top = ranking.slice(0, MAX_BARS);
  const guess = ranking[0]?.id;
  const actualNext = sel + 1 < T ? result.ids[sel + 1] : null;

  const tabBtn = (active: boolean) =>
    `rounded px-2 py-0.5 text-xs font-semibold transition ${
      active ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
    }`;

  return (
    <div className="space-y-6">
      {step === 0 && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
          {s.untrainedHint}
        </p>
      )}

      {/* input */}
      <div>
        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
          {s.inputLabel}
        </label>
        <input
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setPos(null);
          }}
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
        />
        <p className="mt-2 text-[11px] text-slate-400">{s.clickHint}</p>
        <div className="mt-2 flex flex-wrap gap-1">
          {result.ids.map((id, i) => (
            <button
              key={i}
              onClick={() => setPos(i)}
              className={`rounded px-1.5 py-1 font-mono text-sm transition ${
                i === sel
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {charLabel(itos, id)}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* attention heatmap */}
        <div>
          <h3 className="text-sm font-semibold text-slate-800">{s.attnHeading}</h3>
          <p className="mb-3 text-[11px] text-slate-400">{s.attnHelp}</p>
          <div className="mb-3 flex flex-wrap items-center gap-1">
            <span className="mr-1 text-xs text-slate-500">{s.layerLabel}</span>
            {Array.from({ length: nLayer }, (_, i) => (
              <button key={i} onClick={() => setLayer(i)} className={tabBtn(layerSel === i)}>
                {i + 1}
              </button>
            ))}
            <span className="ml-3 mr-1 text-xs text-slate-500">{s.headLabel}</span>
            {Array.from({ length: nHead }, (_, i) => (
              <button key={i} onClick={() => setHead(i)} className={tabBtn(headSel === i)}>
                {i + 1}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <div
              className="inline-grid gap-0.5"
              style={{ gridTemplateColumns: `auto repeat(${T}, 1.1rem)` }}
            >
              <div />
              {result.ids.map((id, c) => (
                <div key={`h${c}`} className="text-center font-mono text-[10px] text-slate-400">
                  {charLabel(itos, id)}
                </div>
              ))}
              {result.ids.map((rid, r) => (
                <Fragment key={`r${r}`}>
                  <button
                    onClick={() => setPos(r)}
                    className={`pr-1 text-right font-mono text-[10px] ${
                      r === sel ? "font-bold text-indigo-600" : "text-slate-400"
                    }`}
                  >
                    {charLabel(itos, rid)}
                  </button>
                  {result.ids.map((_, c) => {
                    const future = c > r;
                    const w = future ? 0 : (view.weights[r * T + c] ?? 0);
                    return (
                      <div
                        key={`c${r}-${c}`}
                        title={`${charLabel(itos, rid)} → ${charLabel(itos, result.ids[c])}: ${(
                          w * 100
                        ).toFixed(0)}%`}
                        className={`h-[1.1rem] w-[1.1rem] rounded-sm ${
                          r === sel ? "ring-1 ring-indigo-400" : ""
                        }`}
                        style={{
                          backgroundColor: future
                            ? "#f1f5f9"
                            : `rgba(79,70,229,${(0.08 + 0.92 * w).toFixed(3)})`,
                        }}
                      />
                    );
                  })}
                </Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* next-character probabilities */}
        <div>
          <h3 className="text-sm font-semibold text-slate-800">{s.probHeading}</h3>
          <p className="mb-3 text-[11px] text-slate-400">{s.probHelp}</p>
          <div className="space-y-1">
            {top.map(({ id, p }, rank) => (
              <div key={id} className="flex items-center gap-2">
                <span className="w-6 text-right font-mono text-xs text-slate-500">
                  {charLabel(itos, id)}
                </span>
                <div className="h-4 flex-1 overflow-hidden rounded bg-slate-100">
                  <div
                    className={`h-4 rounded ${rank === 0 ? "bg-indigo-600" : "bg-indigo-400"}`}
                    style={{ width: `${(p * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs tabular-nums text-slate-500">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-sm text-slate-600">
            {actualNext === null ? (
              s.noNext
            ) : (
              <>
                {s.fasitLabel} {s.fasitNext(charLabel(itos, actualNext))}{" "}
                {guess === actualNext ? (
                  <span className="font-semibold text-emerald-600">{s.correct}</span>
                ) : (
                  <span className="font-semibold text-rose-500">{s.wrong}</span>
                )}
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
