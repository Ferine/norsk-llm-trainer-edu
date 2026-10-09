import type { Strings } from "@/lib/i18n";
import { Gloss } from "@/components/Gloss";

// Tapet på prøveteksten (held-out) ved eit gitt treningssteg.
export interface HeldoutPoint {
  step: number;
  loss: number;
}

interface Props {
  data: number[];
  loss: Strings["loss"];
  // Valfritt: prøvetap og vokabularstorleik. Utan dei (t.d. DPO-grafen i
  // RLHF-steget) blir grafen som før – éi linje og siste tap.
  heldout?: HeldoutPoint[];
  vocab?: number;
}

// Kor mykje prøvetapet må ha stige over botnen sin før vi seier det høgt.
const OVERFIT_RISE = 0.15;
// Øvingstapet er støyete frå minibatch til minibatch; readouten viser snittet.
const SMOOTH = 50;

const nf = new Intl.NumberFormat("nb-NO", { maximumFractionDigits: 1 });
// Tap → «vel mellom ca. N teikn» (perpleksitet = e^tap).
const choices = (l: number) => nf.format(Math.exp(l));

function trainingAverage(data: number[], step: number): number {
  const tail = data.slice(Math.max(0, step - SMOOTH), step);
  return tail.reduce((a, b) => a + b, 0) / tail.length;
}

// Enkel, sjølv-teikna SVG-graf over tapet (loss) under trening.
export default function LossChart({ data, loss, heldout = [], vocab }: Props) {
  const W = 640;
  const H = 220;
  const padL = 40;
  const padR = 12;
  const padT = 14;
  const padB = 26;

  if (data.length < 2) {
    return (
      <div className="flex h-[220px] w-full items-center justify-center rounded-[3px] border-2 border-dashed border-rute bg-white/60 px-4 text-center font-mono text-xs text-blyant">
        {loss.empty}
      </div>
    );
  }

  // nedsample for raskare teikning; punkt i ligg ved steg i+1
  const maxPts = 240;
  const every = Math.max(1, Math.ceil(data.length / maxPts));
  const pts: { step: number; v: number }[] = [];
  for (let i = 0; i < data.length; i += every) pts.push({ step: i + 1, v: data[i] });
  if (pts[pts.length - 1].step !== data.length) pts.push({ step: data.length, v: data[data.length - 1] });

  const chance = vocab ? Math.log(vocab) : null;
  const steps = Math.max(data.length, heldout.length ? heldout[heldout.length - 1].step : 0);

  const min = 0;
  const maxV = Math.max(...pts.map((p) => p.v), ...heldout.map((p) => p.loss), chance ?? 0, 0.5);
  const xAt = (step: number) => padL + (step / steps) * (W - padL - padR);
  const yAt = (v: number) => padT + (1 - (v - min) / (maxV - min)) * (H - padT - padB);
  const path = (ps: { step: number; v: number }[]) =>
    ps.map((p, i) => `${i === 0 ? "M" : "L"}${xAt(p.step).toFixed(1)},${yAt(p.v).toFixed(1)}`).join(" ");

  const line = path(pts);
  const area = `${line} L${xAt(steps).toFixed(1)},${H - padB} L${xAt(pts[0].step).toFixed(1)},${H - padB} Z`;
  const heldLine = path(heldout.map((p) => ({ step: p.step, v: p.loss })));

  // Botnen på prøvekurva: der modellen var best på ny tekst.
  let best: HeldoutPoint | null = null;
  for (const p of heldout) if (!best || p.loss < best.loss) best = p;
  const lastHeld = heldout.length ? heldout[heldout.length - 1] : null;
  const trainNow = trainingAverage(data, data.length);
  // A rising test loss alone can also mean unstable training. Only describe
  // overfitting when the smoothed training loss improved since the test minimum.
  const overfit =
    best && lastHeld && best.step > 0 && lastHeld.loss - best.loss >= OVERFIT_RISE &&
    trainNow < trainingAverage(data, best.step) ? best : null;

  const last = data[data.length - 1];

  return (
    <div className="w-full rounded-[3px] border-2 border-blekk bg-white p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" fontFamily="IBM Plex Mono, monospace">
        {/* rutenett – som millimeterpapir */}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = padT + f * (H - padT - padB);
          const val = maxV - f * (maxV - min);
          return (
            <g key={f}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#d7e2ea" strokeWidth={1} />
              <text x={6} y={y + 4} fontSize={10} fill="#5b6672">
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}
        {/* blyantstrek: kor høgt tapet er når ein gjettar heilt på måfå */}
        {chance !== null && (
          <g>
            <line
              x1={padL}
              y1={yAt(chance)}
              x2={W - padR}
              y2={yAt(chance)}
              stroke="#5b6672"
              strokeWidth={1.25}
              strokeDasharray="2 4"
            />
            <text x={W - padR} y={yAt(chance) - 5} fontSize={10} fill="#5b6672" textAnchor="end">
              {loss.chanceMark}
            </text>
          </g>
        )}
        {/* rettepennens strek: elevens feil, ført med rød penn */}
        <path d={area} fill="rgba(196,58,42,0.06)" />
        <path d={line} fill="none" stroke="#c43a2a" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
        {/* prøva: same penn, stipla – ny tekst modellen aldri har øvd på */}
        {heldout.length > 1 && (
          <path
            d={heldLine}
            fill="none"
            stroke="#c43a2a"
            strokeWidth={2.75}
            strokeDasharray="7 4"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
        {overfit && (
          <g>
            <circle cx={xAt(overfit.step)} cy={yAt(overfit.loss)} r={5} fill="#fff" stroke="#c43a2a" strokeWidth={2} />
            <text
              x={xAt(overfit.step)}
              y={yAt(overfit.loss) - 11}
              fontSize={10}
              fill="#c43a2a"
              textAnchor={xAt(overfit.step) > W * 0.8 ? "end" : "middle"}
            >
              {loss.bestMark}
            </text>
          </g>
        )}
        <text x={W - padR} y={H - 8} fontSize={10} fill="#5b6672" textAnchor="end">
          {loss.axisStep}
        </text>
        <text x={padL} y={H - 8} fontSize={10} fill="#5b6672">
          {loss.axisLoss}
        </text>
      </svg>
      {vocab ? (
        <div className="mt-2 space-y-1 px-1 font-mono text-xs text-blyant">
          <LegendRow swatch="solid" label={loss.trainLabel} value={trainNow} note={loss.choosing(choices(trainNow))} />
          {lastHeld && (
            <LegendRow
              swatch="dashed"
              label={loss.heldoutLabel}
              value={lastHeld.loss}
              note={loss.choosing(choices(lastHeld.loss))}
            />
          )}
          {chance !== null && (
            <LegendRow swatch="pencil" label={loss.chanceLabel} value={chance} note={loss.chanceChoosing(vocab)} />
          )}
          {overfit && (
            <p className="pt-1 font-sans text-sm leading-relaxed text-rettepenn" role="status">
              <Gloss text={loss.overfitNote(overfit.step)} />
            </p>
          )}
        </div>
      ) : (
        <div className="mt-1 flex items-center justify-between px-1 font-mono text-xs text-blyant">
          <span>{loss.last} <span className="font-semibold text-rettepenn">{last.toFixed(4)}</span></span>
          <span>{loss.count(data.length)}</span>
        </div>
      )}
    </div>
  );
}

function LegendRow({
  swatch,
  label,
  value,
  note,
}: {
  swatch: "solid" | "dashed" | "pencil";
  label: string;
  value: number;
  note: string;
}) {
  const stroke = swatch === "pencil" ? "#5b6672" : "#c43a2a";
  const width = swatch === "dashed" ? 2.75 : swatch === "solid" ? 1.5 : 1.25;
  const dash = swatch === "dashed" ? "7 4" : swatch === "pencil" ? "2 4" : undefined;
  return (
    <div className="flex items-baseline gap-2">
      <svg width={28} height={8} aria-hidden className="shrink-0">
        <line x1={1} y1={4} x2={27} y2={4} stroke={stroke} strokeWidth={width} strokeDasharray={dash} strokeLinecap="round" />
      </svg>
      <span>
        {label}{" "}
        <span className={`font-semibold ${swatch === "pencil" ? "text-blyant" : "text-rettepenn"}`}>{value.toFixed(2)}</span>{" "}
        · {note}
      </span>
    </div>
  );
}
