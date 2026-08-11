import LossChart from "@/components/LossChart";
import { Card, Advanced } from "@/components/ui";
import { Gloss } from "@/components/Gloss";
import type { useRlhf } from "@/lib/useRlhf";
import type { Strings } from "@/lib/i18n";

type RlhfApi = ReturnType<typeof useRlhf>;

function PrefCard({
  text,
  onPick,
  disabled,
  answerLabel,
  betterLabel,
}: {
  text: string;
  onPick: () => void;
  disabled: boolean;
  answerLabel: string;
  betterLabel: string;
}) {
  return (
    <div className="flex flex-col rounded-[3px] border-2 border-blekk bg-white p-4">
      <div className="etikett mb-2">{answerLabel}</div>
      <p className="min-h-16 flex-1 whitespace-pre-wrap font-mono text-sm text-blekk">{text || "…"}</p>
      <button onClick={onPick} disabled={disabled} className="knapp knapp-blekk mt-3">
        {betterLabel}
      </button>
    </div>
  );
}

export default function Rlhf({
  rlhf,
  examples,
  s,
  // Nullstillinga rullar vektene tilbake, så ho går gjennom App, som spør først.
  onResetTuning,
}: {
  rlhf: RlhfApi;
  examples: string[];
  s: Strings;
  onResetTuning: () => void;
}) {
  const busy = rlhf.baseRunning || rlhf.dpoRunning || rlhf.generating;

  if (!rlhf.started) {
    return (
      <Card className="space-y-4">
        <p className="text-sm leading-relaxed text-blyant"><Gloss text={s.rlhf.introCard} /></p>
        <button onClick={rlhf.start} disabled={rlhf.baseRunning} className="knapp knapp-blekk">
          {s.rlhf.startBtn}
        </button>
      </Card>
    );
  }

  return (
    <Card className="space-y-5">
      {/* lærerens hånd: i flyten (ikkje absolutt) så panelramma aldri kryssar skrifta */}
      <div aria-hidden className="-mb-4 -mt-1 text-right">
        <span className="handnotat text-xl">{s.rlhf.teacherNote}</span>
      </div>

      {rlhf.untrainedHint && (
        <div className="border-l-4 border-rettepenn bg-white px-3 py-2 text-sm leading-relaxed">
          {s.rlhf.untrainedHint}
        </div>
      )}

      <div>
        <label className="etikett mb-1 block">{s.rlhf.startTextLabel}</label>
        <textarea
          value={rlhf.prompt}
          onChange={(e) => rlhf.setPrompt(e.target.value)}
          rows={2}
          className="felt resize-none"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {examples.map((ex) => (
            <button
              key={ex}
              onClick={() => rlhf.setPrompt(ex)}
              className="rounded-[2px] border border-blekk/40 bg-white px-3 py-1 font-mono text-xs text-blekk transition hover:bg-papir"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
        <div>
          <label className="etikett mb-1 block">{s.rlhf.creativity(rlhf.temp.toFixed(2))}</label>
          <input
            type="range"
            min={0.3}
            max={1.5}
            step={0.05}
            value={rlhf.temp}
            onChange={(e) => rlhf.setTemp(Number(e.target.value))}
            className="w-full"
          />
        </div>
        <button onClick={rlhf.generatePair} disabled={busy} className="knapp knapp-omriss">
          {rlhf.generating ? s.rlhf.makingPair : s.rlhf.generatePair}
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <PrefCard text={rlhf.pairA?.text ?? ""} onPick={() => rlhf.choose("A")} disabled={busy} answerLabel={s.rlhf.prefAnswer("A")} betterLabel={s.rlhf.prefBetter("A")} />
        <PrefCard text={rlhf.pairB?.text ?? ""} onPick={() => rlhf.choose("B")} disabled={busy} answerLabel={s.rlhf.prefAnswer("B")} betterLabel={s.rlhf.prefBetter("B")} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button onClick={rlhf.skip} disabled={busy} className="knapp knapp-omriss knapp-sm">
          {s.rlhf.skip}
        </button>
        {!rlhf.dpoRunning ? (
          <button
            onClick={rlhf.trainMore}
            disabled={rlhf.baseRunning || rlhf.metrics.count === 0}
            className="knapp knapp-blekk knapp-sm"
          >
            {s.rlhf.trainMore}
          </button>
        ) : (
          <button onClick={rlhf.stopTrainMore} className="knapp knapp-rettepenn knapp-sm">
            {s.rlhf.stop}
          </button>
        )}
        <button onClick={onResetTuning} disabled={busy} className="knapp knapp-omriss knapp-sm">
          {s.rlhf.resetTuning}
        </button>
        {/* talet på val er synleg – det forklarer kvifor «Tren mer» er grått ved 0 */}
        <span className="ml-auto font-mono text-xs text-blyant">
          {s.rlhf.prefs} <b className="text-blekk">{rlhf.metrics.count}</b>
        </span>
      </div>

      <Advanced label={s.rlhf.statsLabel}>
        <div className="space-y-3">
          <div className="flex flex-wrap gap-4 font-mono text-xs text-blyant">
            <span>
              {s.rlhf.margin} <b className="text-blekk">{rlhf.metrics.margin.toFixed(3)}</b>
            </span>
            <span>
              {s.rlhf.winRate} <b className="text-blekk">{(rlhf.metrics.winRate * 100).toFixed(0)}%</b>
            </span>
          </div>
          <div>
            <h3 className="mb-2 font-semibold text-blekk">{s.rlhf.dpoLossHeading}</h3>
            <LossChart data={rlhf.losses} loss={s.loss} />
            <p className="mt-2 text-xs leading-relaxed text-blyant"><Gloss text={s.rlhf.dpoHelp} /></p>
          </div>
        </div>
      </Advanced>
    </Card>
  );
}
