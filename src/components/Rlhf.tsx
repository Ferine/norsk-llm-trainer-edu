import LossChart from "@/components/LossChart";
import { Card } from "@/components/ui";
import type { useRlhf } from "@/lib/useRlhf";

type RlhfApi = ReturnType<typeof useRlhf>;

function PrefCard({
  label,
  text,
  onPick,
  disabled,
}: {
  label: string;
  text: string;
  onPick: () => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-4">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Svar {label}</div>
      <p className="min-h-16 flex-1 whitespace-pre-wrap font-mono text-sm text-slate-700">{text || "…"}</p>
      <button
        onClick={onPick}
        disabled={disabled}
        className="mt-3 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
      >
        👍 {label} er betre
      </button>
    </div>
  );
}

export default function Rlhf({ rlhf, examples }: { rlhf: RlhfApi; examples: string[] }) {
  const busy = rlhf.baseRunning || rlhf.dpoRunning || rlhf.generating;

  if (!rlhf.started) {
    return (
      <Card className="space-y-4">
        <p className="text-sm text-slate-600">
          RLHF («Reinforcement Learning from Human Feedback») lærer modellen kva slags svar vi
          menneske føretrekkjer. Vi viser deg to framhald, du vel det beste, og modellen blir
          justert mot valet ditt – forankra til ein frosen referansemodell (DPO).
        </p>
        <button
          onClick={rlhf.start}
          disabled={rlhf.baseRunning}
          className="rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-violet-200 transition hover:bg-violet-500 disabled:opacity-50"
        >
          Start preferanse-trening
        </button>
      </Card>
    );
  }

  return (
    <Card className="space-y-5">
      {rlhf.untrainedHint && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          Tips: tren modellen først i steg 3 – då blir framhalda meir meiningsfulle.
        </div>
      )}

      <div>
        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">Starttekst</label>
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
            Kreativitet: {rlhf.temp.toFixed(2)}
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
          {rlhf.generating ? "Lagar par…" : "↻ Generer eit par"}
        </button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <PrefCard label="A" text={rlhf.pairA?.text ?? ""} onPick={() => rlhf.choose("A")} disabled={busy} />
        <PrefCard label="B" text={rlhf.pairB?.text ?? ""} onPick={() => rlhf.choose("B")} disabled={busy} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={rlhf.skip}
          disabled={busy}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
        >
          Hopp over (likeverdige)
        </button>
        {!rlhf.dpoRunning ? (
          <button
            onClick={rlhf.trainMore}
            disabled={rlhf.baseRunning || rlhf.metrics.count === 0}
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
          >
            Tren meir på preferansane
          </button>
        ) : (
          <button
            onClick={rlhf.stopTrainMore}
            className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-500"
          >
            ⏸ Stopp
          </button>
        )}
        <button
          onClick={rlhf.resetTuning}
          disabled={busy}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-50 disabled:opacity-50"
        >
          ↺ Nullstill justering
        </button>
        <div className="ml-auto flex gap-3 text-xs text-slate-500">
          <span>
            Preferansar: <b className="text-slate-800">{rlhf.metrics.count}</b>
          </span>
          <span>
            Margin: <b className="text-slate-800">{rlhf.metrics.margin.toFixed(3)}</b>
          </span>
          <span>
            Vinnar-rate: <b className="text-slate-800">{(rlhf.metrics.winRate * 100).toFixed(0)}%</b>
          </span>
        </div>
      </div>

      <div>
        <h3 className="mb-2 font-semibold text-slate-900">DPO-tap over tid</h3>
        <LossChart data={rlhf.losses} />
        <p className="mt-2 text-xs text-slate-500">
          Margin = kor mykje meir sannsynleg det valde framhaldet er enn det avviste, samanlikna med
          referansemodellen. Høgare margin og vinnar-rate = modellen følgjer preferansane dine.
        </p>
      </div>
    </Card>
  );
}
