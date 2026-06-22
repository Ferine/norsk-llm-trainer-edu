import type { Strings } from "@/lib/i18n";

interface Props {
  data: number[];
  loss: Strings["loss"];
}

// Enkel, sjølv-teikna SVG-graf over tapet (loss) under trening.
export default function LossChart({ data, loss }: Props) {
  const W = 640;
  const H = 220;
  const padL = 40;
  const padR = 12;
  const padT = 14;
  const padB = 26;

  if (data.length < 2) {
    return (
      <div className="flex h-[220px] w-full items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 text-sm text-slate-400">
        {loss.empty}
      </div>
    );
  }

  // nedsample for raskare teikning
  const maxPts = 240;
  const step = Math.max(1, Math.ceil(data.length / maxPts));
  const pts: number[] = [];
  for (let i = 0; i < data.length; i += step) pts.push(data[i]);
  if (pts[pts.length - 1] !== data[data.length - 1]) pts.push(data[data.length - 1]);

  const min = 0;
  const maxV = Math.max(...pts, 0.5);
  const xAt = (i: number) => padL + (i / (pts.length - 1)) * (W - padL - padR);
  const yAt = (v: number) => padT + (1 - (v - min) / (maxV - min)) * (H - padT - padB);

  const line = pts.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");
  const area = `${line} L${xAt(pts.length - 1).toFixed(1)},${H - padB} L${xAt(0).toFixed(1)},${H - padB} Z`;

  const last = pts[pts.length - 1];

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="lossFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* rutenett */}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = padT + f * (H - padT - padB);
          const val = maxV - f * (maxV - min);
          return (
            <g key={f}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#e2e8f0" strokeWidth={1} />
              <text x={6} y={y + 4} fontSize={10} fill="#94a3b8">
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}
        <path d={area} fill="url(#lossFill)" />
        <path d={line} fill="none" stroke="#4f46e5" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <text x={W - padR} y={H - 8} fontSize={10} fill="#94a3b8" textAnchor="end">
          {loss.axisStep}
        </text>
        <text x={padL} y={H - 8} fontSize={10} fill="#94a3b8">
          {loss.axisLoss}
        </text>
      </svg>
      <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
        <span>{loss.last} <span className="font-semibold text-indigo-600">{last.toFixed(4)}</span></span>
        <span>{loss.count(data.length)}</span>
      </div>
    </div>
  );
}
