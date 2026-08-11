import { useCallback, useEffect, useState } from "react";
import {
  cloneTransformer,
  evalLoss,
  generate,
  mulberry32,
  quantizeFfnMxfp4,
  type QuantStats,
  type Transformer,
} from "@/lib/ml";
import { Utskrift } from "@/components/ui";
import { Gloss } from "@/components/Gloss";
import type { buildTokenizer } from "@/lib/corpus";
import type { Strings } from "@/lib/i18n";

// «Modellen på slankekur»: same triks som Kimi K3 gjer før modellen blir sett i
// drift (§4.1.4) – dei store matrisene får berre 4 bit per tal. Vi måler på ein
// kopi, så ingen mistar den trente modellen sin, og viser båe to skrive.

interface Engine {
  model: Transformer;
  data: number[];
  tokenizer: ReturnType<typeof buildTokenizer>;
}

interface Props {
  getEngine: () => Engine | null;
  step: number;
  engineGen: number;
  prompt: string;
  locale: string;
  s: Strings["train"]["slank"];
  /** Seier frå når slankekuren er køyrd, så rekneark-eksporten kan ta med arket. */
  onRan?: () => void;
}

interface Result {
  stats: QuantStats;
  lossFull: number;
  lossQuant: number;
  textFull: string;
  textQuant: string;
  // startteksten desse to utskriftene faktisk vart skrivne frå
  seed: string;
  at: number;
}

const EVAL_BATCHES = 16;
const EVAL_SEED = 4711;
const SAMPLE_LEN = 60;

export default function Slankekur({
  getEngine,
  step,
  engineGen,
  prompt,
  locale,
  s,
  onRan,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<Result | null>(null);

  // Ny motor eller ny trening: gamle måltal seier ikkje lenger noko sant.
  useEffect(() => setRes(null), [engineGen]);

  const run = useCallback(() => {
    const eng = getEngine();
    if (!eng || step === 0 || busy) return;
    setBusy(true);
    // Gje nettlesaren ein frame til å teikna knappen før vi blokkerer tråden.
    window.setTimeout(() => {
      const { model, data, tokenizer } = eng;
      const seqLen = model.seqLen;
      const greedy = { temperature: 0, topK: 1, length: SAMPLE_LEN };

      const lossFull = evalLoss(model, data, seqLen, EVAL_BATCHES, mulberry32(EVAL_SEED));
      const textFull = generate(
        model,
        tokenizer.decode,
        tokenizer.encode,
        prompt,
        greedy,
        mulberry32(EVAL_SEED)
      );

      const copy = cloneTransformer(model);
      const stats = quantizeFfnMxfp4(copy);
      // Same utdrag og same start som over – berre vektene er ulike.
      const lossQuant = evalLoss(copy, data, seqLen, EVAL_BATCHES, mulberry32(EVAL_SEED));
      const textQuant = generate(
        copy,
        tokenizer.decode,
        tokenizer.encode,
        prompt,
        greedy,
        mulberry32(EVAL_SEED)
      );

      setRes({ stats, lossFull, lossQuant, textFull, textQuant, seed: prompt, at: step });
      setBusy(false);
      onRan?.();
    }, 32);
  }, [busy, getEngine, onRan, prompt, step]);

  const kb = (bytes: number) =>
    `${(bytes / 1024).toLocaleString(locale, { maximumFractionDigits: 1 })} kB`;

  return (
    <div className="space-y-3">
      <p className="text-xs leading-relaxed text-blyant"><Gloss text={s.help} /></p>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={run}
          disabled={step === 0 || busy}
          className="knapp knapp-omriss knapp-sm disabled:opacity-50"
        >
          {busy ? s.busy : s.run}
        </button>
        {step === 0 && <span className="font-mono text-[11px] text-blyant">{s.idle}</span>}
        {res && <span className="font-mono text-[11px] text-blyant">{s.measuredAt(res.at)}</span>}
      </div>

      {res && (
        <>
          <table className="w-full border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-blekk/30 text-left">
                <th className="py-1 pr-3 font-normal text-blyant" />
                <th className="py-1 pr-3 font-semibold text-blekk">{s.colFull}</th>
                <th className="py-1 font-semibold text-blekk">{s.colQuant}</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-marg">
                <td className="py-1.5 pr-3 text-blyant">{s.rowSize}</td>
                <td className="py-1.5 pr-3 text-blekk">{kb(res.stats.bytesBefore)}</td>
                <td className="py-1.5 text-blekk">
                  {kb(res.stats.bytesAfter)}{" "}
                  <span className="text-blyant">
                    {s.shrink(
                      (res.stats.bytesBefore / res.stats.bytesAfter).toLocaleString(locale, {
                        maximumFractionDigits: 1,
                      })
                    )}
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-1.5 pr-3 text-blyant">{s.rowLoss}</td>
                <td className="py-1.5 pr-3 font-semibold text-rettepenn">
                  {res.lossFull.toFixed(3)}
                </td>
                <td className="py-1.5 font-semibold text-rettepenn">{res.lossQuant.toFixed(3)}</td>
              </tr>
            </tbody>
          </table>

          <div className="tavle space-y-2 p-4">
            <div>
              <div className="etikett text-kritt/70">{s.colFull}</div>
              <p className="whitespace-pre-wrap font-mono text-sm text-kritt">
                <Utskrift text={res.textFull} seed={res.seed} />
              </p>
            </div>
            <div>
              <div className="etikett text-kritt/70">{s.colQuant}</div>
              <p className="whitespace-pre-wrap font-mono text-sm text-kritt">
                <Utskrift text={res.textQuant} seed={res.seed} />
              </p>
            </div>
          </div>

          <p className="text-xs leading-relaxed text-blyant"><Gloss text={s.note} /></p>
        </>
      )}
    </div>
  );
}
