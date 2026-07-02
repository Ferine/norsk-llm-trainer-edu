import { useCallback, useEffect, useRef } from "react";

// «Skruane blir vridde»: eit stridt-utval av modellens vekter, teikna som
// ruter der mørk blekkfarge = stor endring sidan førre måling. Målt i praksis
// (preset «liten», frø 42/1337): snittendringa per steg er ~0,2·lr, og delen
// ruter med kraftig vriding fell frå ~8 % tidleg til ~1 % ved konvergens –
// det er dei mørke rutene som ber historia. SHARPEN skalerer så dei
// dominerer visuelt. Alt er ekte verdiar frå modellen, ikkje ein animasjon.

interface Props {
  getParams: () => { d: Float32Array }[] | null;
  step: number;
  engineGen: number;
  lr: number;
  help: string;
  idleText: string;
}

const SHARPEN = 0.5; // lågare = mørkare bilete; 0.5 gjer dei kraftig vridde rutene tydelege
const COLS = 64;
const ROWS = 18;
const CELL = 12;
const GAP = 2;
const W = COLS * CELL;
const H = ROWS * CELL;

export default function Skruer({ getParams, step, engineGen, lr, help, idleText }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pickRef = useRef<{ t: number; i: number }[]>([]);
  const prevRef = useRef<Float32Array | null>(null);
  const prevStepRef = useRef(0);

  const draw = useCallback((heat: Float32Array) => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
    for (let k = 0; k < heat.length; k++) {
      const x = (k % COLS) * CELL;
      const y = Math.floor(k / COLS) * CELL;
      ctx.fillStyle = `rgba(29,54,82,${(0.05 + 0.95 * heat[k]).toFixed(3)})`;
      ctx.fillRect(x, y, CELL - GAP, CELL - GAP);
    }
  }, []);

  // Ny motor (nullstilling, ny storleik, nytt språk): nytt utval og nullpunkt.
  useEffect(() => {
    const params = getParams();
    if (!params) return;
    const sizes = params.map((p) => p.d.length);
    const total = sizes.reduce((a, b) => a + b, 0);
    const n = COLS * ROWS;
    const pick: { t: number; i: number }[] = [];
    for (let k = 0; k < n; k++) {
      let flat = Math.min(total - 1, Math.floor(((k + 0.5) * total) / n));
      let t = 0;
      while (flat >= sizes[t]) {
        flat -= sizes[t];
        t++;
      }
      pick.push({ t, i: flat });
    }
    pickRef.current = pick;
    const prev = new Float32Array(n);
    pick.forEach((p, k) => {
      prev[k] = params[p.t].d[p.i];
    });
    prevRef.current = prev;
    prevStepRef.current = 0;
    draw(new Float32Array(n)); // alt i ro
  }, [engineGen, getParams, draw]);

  // Kvar treningsbolk: mål kor mykje kvar utvald skrue flytta seg.
  useEffect(() => {
    if (step === 0) return;
    const params = getParams();
    const prev = prevRef.current;
    const pick = pickRef.current;
    if (!params || !prev || pick.length === 0) return;
    const dSteps = Math.max(1, step - prevStepRef.current);
    const scale = lr * dSteps * SHARPEN;
    const heat = new Float32Array(pick.length);
    pick.forEach((p, k) => {
      const w = params[p.t].d[p.i];
      heat[k] = Math.min(1, Math.abs(w - prev[k]) / scale);
      prev[k] = w;
    });
    prevStepRef.current = step;
    draw(heat);
  }, [step, engineGen, lr, getParams, draw]);

  return (
    <div className="space-y-2">
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        role="img"
        aria-label={help}
        className="h-auto w-full rounded-[3px] border-2 border-blekk bg-white"
      />
      <p className="text-xs leading-relaxed text-blyant">{step === 0 ? idleText : help}</p>
    </div>
  );
}
