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
            className="knapp knapp-blekk"
          >
            {s.mergeBtn}
          </button>
          <button onClick={() => setApplied(0)} className="knapp knapp-omriss">
            {s.resetBtn}
          </button>
          <span className="ml-1 font-mono text-xs tabular-nums text-blyant">{s.mergeCount(k, n)}</span>
        </div>
        <input
          type="range"
          min={0}
          max={n}
          value={k}
          onChange={(e) => setApplied(Number(e.target.value))}
          disabled={n === 0}
          className="w-full"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* this merge */}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-blekk">{s.thisMergeHeading}</h3>
          {current === null ? (
            <p className="text-sm text-blyant">{s.noMergeYet}</p>
          ) : (
            <div className="rounded-[3px] border-2 border-blekk bg-rute/20 p-4">
              <div className="flex items-center justify-center gap-2 font-mono text-lg">
                <span className="brikke px-2 py-1">{current.a}</span>
                <span className="text-blyant">+</span>
                <span className="brikke px-2 py-1">{current.b}</span>
                <span className="text-blyant">→</span>
                <span className="brikke bg-tusj px-2 py-1 font-semibold">{current.merged}</span>
              </div>
              <p className="mt-2 text-center text-sm text-blekk">{s.foundTimes(current.count)}</p>
              {current.rivals.length > 0 && (
                <p className="mt-1 text-center font-mono text-xs text-blyant">
                  {s.rivalsLabel} {current.rivals.map((r) => `${r.pair} (${r.count})`).join(", ")}
                </p>
              )}
            </div>
          )}
        </div>

        {/* rules so far */}
        <div>
          <h3 className="mb-2 text-sm font-semibold text-blekk">{s.rulesHeading}</h3>
          <div className="max-h-44 overflow-auto rounded-[3px] border-2 border-blekk bg-white p-2">
            {rules.length === 0 ? (
              <p className="p-2 text-sm text-blyant">{s.noRules}</p>
            ) : (
              <ol className="space-y-1">
                {rules.map((m) => (
                  <li
                    key={m.rank}
                    className="flex items-center gap-2 font-mono text-xs text-blyant"
                  >
                    <span className="w-5 text-right text-blyant/60">{m.rank + 1}.</span>
                    <span>
                      {m.a}+{m.b}
                    </span>
                    <span className="text-blyant/60">→</span>
                    <span className="font-semibold text-blekk">{m.merged}</span>
                    <span className="ml-auto text-blyant/60">{m.count}×</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      </div>

      {/* sample sentence re-tokenized */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-blekk">
          {s.sentenceHeading(tokens.length, baseCount)}
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`brikke px-2 py-1 text-sm ${t.length > 1 ? "bg-tusj font-semibold" : ""}`}
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* vocab tally + payoff */}
      <p className="font-mono text-xs text-blyant">{s.vocabLine(baseVocab.length, k, vocabSize)}</p>
      <p className="border-l-4 border-rettepenn bg-white px-3 py-2 text-sm leading-relaxed">
        <b className="font-mono text-xs font-semibold uppercase tracking-wider text-rettepenn">NB! </b>
        {s.payoff}
      </p>
    </div>
  );
}
