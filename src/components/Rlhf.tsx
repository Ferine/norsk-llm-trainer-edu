import LossChart from "@/components/LossChart";
import { Card } from "@/components/ui";
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
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-4">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{answerLabel}</div>
      <p className="min-h-16 flex-1 whitespace-pre-wrap font-mono text-sm text-slate-700">{text || "…"}</p>
      <button
        onClick={onPick}
        disabled={disabled}
        className="mt-3 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
      >
        {betterLabel}
      </button>
    </div>
  );
}

export default function Rlhf({ rlhf, examples, s }: { rlhf: RlhfApi; examples: string[]; s: Strings }) {
  const busy = rlhf.baseRunning || rlhf.dpoRunning || rlhf.generating;

  if (!rlhf.started) {
    return (
      <Card className="space-y-4">
        <p className="text-sm text-slate-600">
          {s.rlhf.introCard}
        </p>
        <button
          onClick={rlhf.start}
          disabled={rlhf.baseRunning}
          className="rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-violet-200 transition hover:bg-violet-500 disabled:opacity-50"
        >
          {s.rlhf.startBtn}
        </button>
      </Card>
    );
  }

  return (
    <Card className="space-y-5">
      {rlhf.untrainedHint && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          {s.rlhf.untrainedHint}
        </div>
      )}

      <div>
        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">{s.rlhf.startTextLabel}</label>
        <textarea
          value={rlhf.prompt}
          onChange={(e) => rlhf.setPrompt(e.target.value)}
          rows={2}
          className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {examples.map((ex) => (
            <button
              key={ex}
              onClick={() => rlhf.setPrompt(ex)}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 transition hover:border-indigo-300 hover:text-indigo-600"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-[1fr_auto] sm:items-end">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            {s.rlhf.creativity(rlhf.temp.toFixed(2))}
          </label>
          <input
            type="range"
            min={0.3}
            max={1.5}
            step={0.05}
            value={rlhf.temp}
            onChange={(e) => rlhf.setTemp(Number(e.target.value))}
            className="w-full accent-indigo-600"
          />
        </div>
        <button
          onClick={rlhf.generatePair}
          disabled={busy}
          className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
        >
          {rlhf.generating ? s.rlhf.makingPair : s.rlhf.generatePair}
        </button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <PrefCard text={rlhf.pairA?.text ?? ""} onPick={() => rlhf.choose("A")} disabled={busy} answerLabel={s.rlhf.prefAnswer("A")} betterLabel={s.rlhf.prefBetter("A")} />
        <PrefCard text={rlhf.pairB?.text ?? ""} onPick={() => rlhf.choose("B")} disabled={busy} answerLabel={s.rlhf.prefAnswer("B")} betterLabel={s.rlhf.prefBetter("B")} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={rlhf.skip}
          disabled={busy}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
        >
          {s.rlhf.skip}
        </button>
        {!rlhf.dpoRunning ? (
          <button
            onClick={rlhf.trainMore}
            disabled={rlhf.baseRunning || rlhf.metrics.count === 0}
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {s.rlhf.trainMore}
          </button>
        ) : (
          <button
            onClick={rlhf.stopTrainMore}
            className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-500"
          >
            {s.rlhf.stop}
          </button>
        )}
        <button
          onClick={rlhf.resetTuning}
          disabled={busy}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
        >
          {s.rlhf.resetTuning}
        </button>
        <div className="ml-auto flex gap-3 text-xs text-slate-500">
          <span>
            {s.rlhf.prefs} <b className="text-slate-800">{rlhf.metrics.count}</b>
          </span>
          <span>
            {s.rlhf.margin} <b className="text-slate-800">{rlhf.metrics.margin.toFixed(3)}</b>
          </span>
          <span>
            {s.rlhf.winRate} <b className="text-slate-800">{(rlhf.metrics.winRate * 100).toFixed(0)}%</b>
          </span>
        </div>
      </div>

      <div>
        <h3 className="mb-2 font-semibold text-slate-900">{s.rlhf.dpoLossHeading}</h3>
        <LossChart data={rlhf.losses} label={s.lossLast} />
        <p className="mt-2 text-xs text-slate-500">
          {s.rlhf.dpoHelp}
        </p>
      </div>
    </Card>
  );
}
