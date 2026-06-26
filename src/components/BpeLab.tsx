import { useEffect, useMemo, useState } from "react";
import { learnBpe, tokenizeSentence } from "@/lib/bpe";
import type { Strings } from "@/lib/i18n";

interface Props {
  corpus: string;
  sampleSentence: string;
  s: Strings["bpe"];
}

const NUM_MERGES = 80;

export default function BpeLab({ corpus, sampleSentence, s }: Props) {
  const { baseVocab, merges } = useMemo(() => learnBpe(corpus, NUM_MERGES), [corpus]);
  const [applied, setApplied] = useState(0);

  // reset the demo when the corpus (language) changes
  useEffect(() => {
    setApplied(0);
  }, [corpus]);

  const n = merges.length;
  const k = Math.min(applied, n);

  const current = k > 0 ? merges[k - 1] : null;
  const rules = merges.slice(0, k);
  const vocabSize = baseVocab.length + k;
  const tokens = useMemo(
    () => tokenizeSentence(sampleSentence, merges, k),
    [sampleSentence, merges, k]
  );
  const baseCount = useMemo(
    () => tokenizeSentence(sampleSentence, merges, 0).length,
    [sampleSentence, merges]
  );

  return (
    <div className="space-y-6">
      {/* controls */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setApplied((a) => Math.min(a + 1, n))}
            disabled={k >= n}
            className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-indigo-200 transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {s.mergeBtn}
          </button>
          <button
            onClick={() => setApplied(0)}
            className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-600 transition hover:bg-slate-50"
          >
            {s.resetBtn}
          </button>
          <span className="ml-1 text-sm tabular-nums text-slate-500">{s.mergeCount(k, n)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={n}
          value={k}
          onChange={(e) => setApplied(Number(e.target.value))}
          disabled={n === 0}
          className="w-full accent-indigo-600"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* this merge */}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-800">{s.thisMergeHeading}</h3>
          {current === null ? (
            <p className="text-sm text-slate-400">{s.noMergeYet}</p>
          ) : (
            <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4">
              <div className="flex items-center justify-center gap-2 font-mono text-lg">
                <span className="rounded bg-white px-2 py-1 text-slate-700">{current.a}</span>
                <span className="text-slate-400">+</span>
                <span className="rounded bg-white px-2 py-1 text-slate-700">{current.b}</span>
                <span className="text-slate-400">→</span>
                <span className="rounded bg-indigo-600 px-2 py-1 font-semibold text-white">
                  {current.merged}
                </span>
              </div>
              <p className="mt-2 text-center text-sm text-indigo-700">{s.foundTimes(current.count)}</p>
              {current.rivals.length > 0 && (
                <p className="mt-1 text-center text-xs text-slate-500">
                  {s.rivalsLabel} {current.rivals.map((r) => `${r.pair} (${r.count})`).join(", ")}
                </p>
              )}
            </div>
          )}
        </div>

        {/* rules so far */}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-800">{s.rulesHeading}</h3>
          <div className="max-h-44 overflow-auto rounded-xl border border-slate-200 bg-white p-2">
            {rules.length === 0 ? (
              <p className="p-2 text-sm text-slate-400">{s.noRules}</p>
            ) : (
              <ol className="space-y-1">
                {rules.map((m) => (
                  <li
                    key={m.rank}
                    className="flex items-center gap-2 font-mono text-xs text-slate-600"
                  >
                    <span className="w-5 text-right text-slate-400">{m.rank + 1}.</span>
                    <span>
                      {m.a}+{m.b}
                    </span>
                    <span className="text-slate-300">→</span>
                    <span className="font-semibold text-indigo-600">{m.merged}</span>
                    <span className="ml-auto text-slate-400">{m.count}×</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      </div>

      {/* sample sentence re-tokenized */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-800">
          {s.sentenceHeading(tokens.length, baseCount)}
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`inline-flex items-center rounded-lg border px-2 py-1 font-mono text-sm ${
                t.length > 1
                  ? "border-indigo-300 bg-indigo-100 font-semibold text-indigo-700"
                  : "border-slate-200 bg-slate-50 text-slate-600"
              }`}
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* vocab tally + payoff */}
      <p className="text-sm text-slate-600">{s.vocabLine(baseVocab.length, k, vocabSize)}</p>
      <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
        💡 {s.payoff}
      </p>
    </div>
  );
}
