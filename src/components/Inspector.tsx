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
    const { logits, attn, routes } = model.inspect(ids);
    return { ids, logits, attn, routes };
  }, [text, model, step, tokenizer]);

  if (!model || !tokenizer || !result) {
    return <p className="text-sm text-blyant">{s.notReady}</p>;
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
  const route = result.routes.find((v) => v.layer === layerSel) ?? result.routes[0];

  const probs = rowProbs(result.logits, sel);
  const ranking = Array.from(probs, (p, id) => ({ id, p })).sort((a, b) => b.p - a.p);
  const top = ranking.slice(0, MAX_BARS);
  const guess = ranking[0]?.id;
  const actualNext = sel + 1 < T ? result.ids[sel + 1] : null;

  const tabBtn = (active: boolean) =>
    `rounded-[2px] border px-2 py-0.5 text-xs font-semibold transition ${
      active ? "border-blekk bg-blekk text-white" : "border-blekk/40 bg-white text-blekk hover:bg-papir"
    }`;

  return (
    <div className="space-y-6">
      {step === 0 && (
        <p className="border-l-4 border-rettepenn bg-white px-3 py-2 text-sm leading-relaxed">
          {s.untrainedHint}
        </p>
      )}

      {/* input */}
      <div>
        <label className="etikett mb-1 block">{s.inputLabel}</label>
        <input
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setPos(null);
          }}
          className="felt font-mono"
        />
        <p className="mt-2 text-[11px] text-blyant">{s.clickHint}</p>
        <div className="mt-2 flex flex-wrap gap-1">
          {result.ids.map((id, i) => (
            <button
              key={i}
              onClick={() => setPos(i)}
              className={`brikke px-1.5 py-1 text-sm transition ${
                i === sel ? "bg-tusj font-semibold" : "hover:bg-papir"
              }`}
            >
              {charLabel(itos, id)}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* attention heatmap */}
        <div>
          <h3 className="text-sm font-semibold text-blekk">{s.attnHeading}</h3>
          <p className="mb-3 text-[11px] text-blyant">{s.attnHelp}</p>
          <div className="mb-3 flex flex-wrap items-center gap-1">
            <span className="mr-1 font-mono text-xs text-blyant">{s.layerLabel}</span>
            {Array.from({ length: nLayer }, (_, i) => (
              <button key={i} onClick={() => setLayer(i)} className={tabBtn(layerSel === i)}>
                {i + 1}
              </button>
            ))}
            <span className="ml-3 mr-1 font-mono text-xs text-blyant">{s.headLabel}</span>
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
                <div key={`h${c}`} className="text-center font-mono text-[10px] text-blyant">
                  {charLabel(itos, id)}
                </div>
              ))}
              {result.ids.map((rid, r) => (
                <Fragment key={`r${r}`}>
                  <button
                    onClick={() => setPos(r)}
                    className={`pr-1 text-right font-mono text-[10px] ${
                      r === sel ? "font-bold text-blekk" : "text-blyant"
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
                        className={`h-[1.1rem] w-[1.1rem] rounded-[2px] ${
                          r === sel ? "ring-1 ring-blekk/50" : ""
                        }`}
                        style={{
                          backgroundColor: future
                            ? "#eceff0"
                            : `rgba(29,54,82,${(0.06 + 0.94 * w).toFixed(3)})`,
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
          <h3 className="text-sm font-semibold text-blekk">{s.probHeading}</h3>
          <p className="mb-3 text-[11px] text-blyant">{s.probHelp}</p>
          <div className="space-y-1">
            {top.map(({ id, p }, rank) => (
              <div key={id} className="flex items-center gap-2">
                <span className="w-6 text-right font-mono text-xs text-blyant">
                  {charLabel(itos, id)}
                </span>
                <div className="h-4 flex-1 overflow-hidden rounded-[2px] border border-blekk/20 bg-rute/25">
                  <div
                    className={`h-full ${rank === 0 ? "bg-blekk" : "bg-blyant/50"}`}
                    style={{ width: `${(p * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-xs tabular-nums text-blyant">
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
          {/* fasiten er lærerens domene: rød penn, både ved rett og galt */}
          <p className="mt-3 border-l-4 border-rettepenn bg-white px-3 py-2 text-sm leading-relaxed">
            {actualNext === null ? (
              s.noNext
            ) : (
              <>
                <b className="text-rettepenn">{s.fasitLabel}</b>{" "}
                {s.fasitNext(charLabel(itos, actualNext))}{" "}
                <span className="font-semibold text-rettepenn">
                  {guess === actualNext ? s.correct : s.wrong}
                </span>
              </>
            )}
          </p>
        </div>
      </div>

      {/* kven rutaren sende kvart teikn til – berre når modellen har ekspertar */}
      {route && (
        <div>
          <h3 className="text-sm font-semibold text-blekk">{s.expertHeading}</h3>
          <p className="mb-3 text-[11px] text-blyant">{s.expertHelp}</p>
          <div className="overflow-x-auto">
            <div
              className="inline-grid gap-0.5"
              style={{ gridTemplateColumns: `auto repeat(${route.experts}, 3rem)` }}
            >
              <div />
              {Array.from({ length: route.experts }, (_, e) => (
                <div key={`eh${e}`} className="text-center font-mono text-[10px] text-blyant">
                  {s.expertLabel(e + 1)}
                </div>
              ))}
              {result.ids.map((rid, r) => (
                <Fragment key={`er${r}`}>
                  <button
                    onClick={() => setPos(r)}
                    className={`pr-1 text-right font-mono text-[10px] ${
                      r === sel ? "font-bold text-blekk" : "text-blyant"
                    }`}
                  >
                    {charLabel(itos, rid)}
                  </button>
                  {Array.from({ length: route.experts }, (_, e) => {
                    const g = route.gates[r * route.experts + e] ?? 0;
                    let woke = false;
                    for (let k = 0; k < route.topK; k++)
                      if (route.chosen[r * route.topK + k] === e) woke = true;
                    return (
                      <div
                        key={`ec${r}-${e}`}
                        title={`${charLabel(itos, rid)} → ${s.expertLabel(e + 1)}: ${(
                          g * 100
                        ).toFixed(0)}%`}
                        className={`flex h-[1.1rem] items-center justify-center rounded-[2px] font-mono text-[9px] tabular-nums ${
                          woke ? "border-2 border-blekk" : "border border-blekk/15"
                        } ${g > 0.5 ? "text-white" : "text-blyant"}`}
                        style={{ backgroundColor: `rgba(29,54,82,${(0.06 + 0.94 * g).toFixed(3)})` }}
                      >
                        {(g * 100).toFixed(0)}
                      </div>
                    );
                  })}
                </Fragment>
              ))}
            </div>
          </div>
          <p className="mt-2 text-[11px] text-blyant">{s.expertShared}</p>
        </div>
      )}
    </div>
  );
}
