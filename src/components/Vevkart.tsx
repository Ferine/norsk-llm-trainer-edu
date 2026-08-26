import { useCallback, useEffect, useRef } from "react";
import type { Transformer } from "@/lib/ml";
import type { Strings } from "@/lib/i18n";
import { Gloss } from "@/components/Gloss";

// Vevkartet: kvar einaste vekt i modellen som piksel, sortert etter anatomien
// – innbygging, så merksemd og breitt lag for kvar blokk, til slutt utgangen.
// Blekkstyrken er storleiken på vekta, så du ser strukturen organisere seg
// medan treninga går: frå jamn tilfeldig grå til band med kvar sine mønster.
//
// Layout-ideen (vekter teikna etter modellens anatomi) er inspirert av
// Sultan-papaganis gguf-visualizer (github.com/Sultan-papagani/gguf-visualizer,
// README: «Do whatever you want with it»), men alt her er skrive frå grunnen:
// 2D-canvas over levande vekter – ingen Three.js, inga GGUF-parsing.

interface Props {
  getModel: () => Transformer | null;
  step: number;
  engineGen: number;
  locale: string;
  t: Strings["train"];
}

const GUTTER = 160; // plass til etikettane i venstre marg
const MAPW = 590; // pikselfeltet
const W = GUTTER + MAPW;
const GAP = 8; // luft mellom banda
const PAD = 4;
const BLEKK: [number, number, number] = [29, 54, 82]; // --color-blekk
const BLYANT = "#5b6672"; // --color-blyant
// Vekter som nettopp flytta seg får markeringstusj (--color-tusj, #ffe68c)
// bak blekket, og gløden klingar av bolk for bolk. Terskelen er relativ:
// tidleg i treninga rører ALT på seg litt, så berre rørsle klart over
// snittet får tusj – full glød ved 4× snittet, kvadrert, og glød under
// GOLV blir ikkje teikna, elles legg småspikarane eit gult slør over alt.
// Sjølvkalibrerande, så Adam og Muon les likt.
const DECAY = 0.85;
const GOLV = 0.12;

interface Band {
  label: string;
  tensors: Float32Array[];
  n: number;
}

function band(label: string, tensors: { d: Float32Array }[]): Band {
  const data = tensors.map((t) => t.d);
  return { label, tensors: data, n: data.reduce((a, d) => a + d.length, 0) };
}

// Modellens anatomi i leserekkjefølgje. Same vandring som params-lista i
// ml.ts, berre gruppert slik arkitekturteikninga i steg 4 fortel historia.
function buildBands(model: Transformer, t: Strings["train"]): Band[] {
  const bands: Band[] = [band(t.vevEmb, [model.tokEmb, model.posEmb])];
  if (model.ngramEmb) bands.push(band(t.vevNgram, [model.ngramEmb]));
  model.blocks.forEach((blk, i) => {
    bands.push(band(t.vevAttn(i + 1), [blk.ln1g, blk.ln1b, blk.Wq, blk.Wk, blk.Wv, blk.Wo]));
    const ffn: { d: Float32Array }[] = [blk.ln2g, blk.ln2b, blk.W1, blk.b1];
    if (blk.Wu && blk.bu) ffn.push(blk.Wu, blk.bu);
    ffn.push(blk.W2, blk.b2);
    if (blk.router) ffn.push(blk.router.W);
    for (const e of blk.routed ?? []) {
      ffn.push(e.W1, e.b1);
      if (e.Wu && e.bu) ffn.push(e.Wu, e.bu);
      ffn.push(e.W2, e.b2);
    }
    bands.push(band(blk.router ? t.vevMoe(i + 1) : t.vevFfn(i + 1), ffn));
  });
  bands.push(band(t.vevOut, [model.lnFg, model.lnFb, model.head]));
  return bands;
}

export default function Vevkart({ getModel, step, engineGen, locale, t }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Førre verdi og glød per vekt, flata ut i bandrekkjefølgje. Ny motor
  // (engineGen) eller ny storleik nullstiller båe.
  const prevRef = useRef<Float32Array | null>(null);
  const heatRef = useRef<Float32Array | null>(null);
  const prevStepRef = useRef(0);
  const genRef = useRef(-1);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const model = getModel();
    if (!canvas || !model) return;

    const bands = buildBands(model, t);
    const total = bands.reduce((a, b) => a + b.n, 0);

    if (genRef.current !== engineGen || prevRef.current?.length !== total) {
      genRef.current = engineGen;
      const prev = new Float32Array(total);
      let j = 0;
      for (const b of bands) for (const d of b.tensors) for (let i = 0; i < d.length; i++) prev[j++] = d[i];
      prevRef.current = prev;
      heatRef.current = new Float32Array(total);
      prevStepRef.current = step;
    }
    const prev = prevRef.current!;
    const heat = heatRef.current!;

    // Mål rørsla sidan førre teikning – men berre når klokka faktisk gjekk.
    if (step > prevStepRef.current) {
      let sumD = 0;
      let j = 0;
      for (const b of bands)
        for (const d of b.tensors)
          for (let i = 0; i < d.length; i++, j++) sumD += Math.abs(d[i] - prev[j]);
      const meanD = sumD / Math.max(1, total);
      if (meanD > 0) {
        const invScale = 1 / (4 * meanD);
        j = 0;
        for (const b of bands)
          for (const d of b.tensors)
            for (let i = 0; i < d.length; i++, j++) {
              const w = d[i];
              const rel = Math.min(1, Math.abs(w - prev[j]) * invScale);
              heat[j] = Math.max(heat[j] * DECAY, rel * rel);
              prev[j] = w;
            }
      }
      prevStepRef.current = step;
    }
    // Under ~120k vekter er det plass til 2×2 pikslar per vekt.
    const p = total > 120_000 ? 1 : 2;
    const cols = Math.floor(MAPW / p);

    let y = PAD;
    const tops = bands.map((b) => {
      const top = y;
      y += Math.ceil(b.n / cols) * p + GAP;
      return top;
    });
    const H = y - GAP + PAD;

    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    // Mjuk skala: alfa = |w| / (|w| + 2·snitt). LayerNorm-forsterkingane (~1)
    // blir nesten svarte – det er ærleg, dei ER store – utan at matrisene
    // (~0,05) drukna i kvitt.
    let sum = 0;
    let cnt = 0;
    const stride = Math.max(1, Math.floor(total / 20_000));
    for (const b of bands)
      for (const d of b.tensors)
        for (let i = 0; i < d.length; i += stride) {
          sum += Math.abs(d[i]);
          cnt++;
        }
    const k = 2 * (sum / Math.max(1, cnt)) || 1;

    const img = ctx.createImageData(MAPW, H);
    img.data.fill(255);
    // Papir → tusj etter glød, så blekk etter storleik oppå: markeringstusj
    // under skrifta, akkurat som i ei ekte kladdebok.
    let flat = 0;
    bands.forEach((b, bi) => {
      let j = 0;
      for (const d of b.tensors)
        for (let i = 0; i < d.length; i++, j++, flat++) {
          const a = Math.abs(d[i]) / (Math.abs(d[i]) + k);
          const h = heat[flat] < GOLV ? 0 : heat[flat];
          const bgG = 255 - 25 * h; // mot #ffe68c
          const bgB = 255 - 115 * h;
          const x0 = (j % cols) * p;
          const y0 = tops[bi] + Math.floor(j / cols) * p;
          const r = 255 - (255 - BLEKK[0]) * a;
          const g = bgG - (bgG - BLEKK[1]) * a;
          const bl = bgB - (bgB - BLEKK[2]) * a;
          for (let dy = 0; dy < p; dy++)
            for (let dx = 0; dx < p; dx++) {
              const idx = ((y0 + dy) * MAPW + x0 + dx) * 4;
              img.data[idx] = r;
              img.data[idx + 1] = g;
              img.data[idx + 2] = bl;
            }
        }
    });
    ctx.putImageData(img, GUTTER, 0);

    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    bands.forEach((b, bi) => {
      ctx.font = "600 10px 'IBM Plex Mono', ui-monospace, monospace";
      ctx.fillStyle = "#1d3652";
      ctx.fillText(b.label, GUTTER - 10, tops[bi]);
      if (Math.ceil(b.n / cols) * p >= 26) {
        ctx.font = "9px 'IBM Plex Mono', ui-monospace, monospace";
        ctx.fillStyle = BLYANT;
        ctx.fillText(b.n.toLocaleString(locale), GUTTER - 10, tops[bi] + 13);
      }
    });
  }, [getModel, t, locale, step, engineGen]);

  // Teiknar òg ved steg 0: den jamne, tilfeldige gråa FØR trening er halve
  // poenget – det er henne banda veks fram or.
  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div className="space-y-2">
      <canvas
        ref={canvasRef}
        width={W}
        height={300}
        role="img"
        aria-label={t.vevHelp}
        className="h-auto w-full rounded-[3px] border-2 border-blekk bg-white [image-rendering:pixelated]"
      />
      <p className="text-xs leading-relaxed text-blyant">
        <Gloss text={step === 0 ? t.vevIdle : t.vevHelp} />
      </p>
    </div>
  );
}
