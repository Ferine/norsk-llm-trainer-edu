import type { Strings } from "@/lib/i18n";

interface Props {
  layers: number;
  heads: number;
  dim: number;
  s: Strings;
}

function Box({
  title,
  sub,
  tone = "slate",
  children,
}: {
  title: string;
  sub?: string;
  tone?: "slate" | "indigo" | "violet" | "emerald" | "amber";
  children?: React.ReactNode;
}) {
  const tones: Record<string, string> = {
    slate: "border-slate-200 bg-white text-slate-700",
    indigo: "border-indigo-200 bg-indigo-50 text-indigo-700",
    violet: "border-violet-200 bg-violet-50 text-violet-700",
    emerald: "border-emerald-200 bg-emerald-50 text-emerald-700",
    amber: "border-amber-200 bg-amber-50 text-amber-700",
  };
  return (
    <div className={`rounded-lg border px-3 py-2 text-center ${tones[tone]}`}>
      <div className="text-xs font-semibold sm:text-sm">{title}</div>
      {sub && <div className="text-[10px] opacity-70 sm:text-xs">{sub}</div>}
      {children}
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-1 text-slate-300">
      <svg width="16" height="18" viewBox="0 0 16 18" fill="none">
        <path d="M8 0v14M3 9l5 5 5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

// Skjematisk teikning av transformator-arkitekturen (GPT-stil).
export default function Architecture({ layers, heads, dim, s }: Props) {
  return (
    <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr]">
      {/* Hovudstraum (venstre/sentralt) */}
      <div className="space-y-0 md:order-2">
        <Box title={s.arch.boxInput.title} sub={s.arch.boxInput.sub} tone="slate" />
        <Arrow />
        <Box title={s.arch.boxEmbedding.title} sub={s.arch.boxEmbedding.sub(dim)} tone="indigo" />
        <Arrow />
        {Array.from({ length: layers }).map((_, i) => (
          <div key={i}>
            <Box title={s.arch.boxBlock.title(i + 1)} sub={s.arch.boxBlock.sub} tone="violet">
              <div className="mt-2 space-y-1">
                <Box title={s.arch.boxAttn.title} sub={s.arch.boxAttn.sub(heads)} tone="violet" />
                <Box title={s.arch.boxFfn.title} sub={s.arch.boxFfn.sub} tone="violet" />
                <div className="text-[10px] opacity-70">{s.arch.residualNote}</div>
              </div>
            </Box>
            <Arrow />
          </div>
        ))}
        <Box title={s.arch.boxFinalNorm.title} sub={s.arch.boxFinalNorm.sub} tone="emerald" />
        <Arrow />
        <Box title={s.arch.boxOutHead.title} sub={s.arch.boxOutHead.sub} tone="amber" />
        <Arrow />
        <Box title={s.arch.boxSoftmax.title} sub={s.arch.boxSoftmax.sub} tone="amber" />
      </div>

      {/* Forklaring (høgre på stor skjerm) */}
      <aside className="md:order-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          <h4 className="mb-2 font-semibold text-slate-800">{s.arch.explainHeading}</h4>
          <ul className="space-y-2">
            {s.arch.explain.map((e, i) => {
              const colors = ["text-indigo-600", "text-violet-600", "text-violet-600", "text-emerald-600", "text-amber-600"];
              return (
                <li key={i}>
                  <b className={colors[i]}>{e.b}</b> {e.t}
                </li>
              );
            })}
          </ul>
        </div>
      </aside>

      {/* plasshaldar for grid på stor skjerm */}
      <div className="hidden md:order-1 md:block" />
    </div>
  );
}
