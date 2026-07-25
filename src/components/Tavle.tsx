import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { cn } from "@/utils/cn";
import {
  blurPx,
  chalkOpacity,
  confToSmudge,
  forcedTier,
  lossToFocus,
  supportsElementTexture,
} from "@/lib/chalk";
import { FRAG, VERT } from "./tavle.glsl";

// Tavla der modellen skriv. Teksten kan teiknast med «krit som ikkje har
// sett seg»: kor uklart eit teikn står, er bunde til eit ekte tal frå
// modellen – anten tapet (heile linja) eller sikkerheita per teikn.
// Utan `gauge` oppfører komponenten seg nøyaktig som tavla gjorde før.

export type Gauge =
  | { kind: "loss"; value: number; vocab: number }
  | { kind: "conf"; conf: Float32Array; promptLen: number };

// Standardklassen er oppgåve 7 sin opphavlege stil. Oppgåve 5 hadde ei anna
// høgd og utan linjeavstand – gjeven via textClassName, slik at kvar tavle
// held fram med å sjå ut som ho gjorde før denne komponenten fanst.
const DEFAULT_TEXT_CLASS =
  "min-h-8 whitespace-pre-wrap font-mono text-sm leading-relaxed text-kritt";

interface Props {
  label: string;
  text: string;
  placeholder: string;
  legend: string;
  summary: string;
  gauge?: Gauge;
  className?: string;
  textClassName?: string;
  // Kall-staden kan trekkje lerretet attende sjølv om nettlesaren støttar
  // tier 2 – t.d. det varme opplæringssteget i oppgåve 5, der eit nytt
  // WebGL-kontekst er for dyrt til å skape kvart bilete.
  noCanvas?: boolean;
  children?: ReactNode;
}

export default function Tavle({
  label,
  text,
  placeholder,
  legend,
  summary,
  gauge,
  className,
  textClassName = DEFAULT_TEXT_CLASS,
  noCanvas = false,
  children,
}: Props) {
  const boardRef = useRef<HTMLParagraphElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [tier2, setTier2] = useState(false);

  useEffect(() => {
    const forced = forcedTier();
    if (forced !== null) {
      setTier2(forced === 2);
      return;
    }
    setTier2(supportsElementTexture());
  }, []);

  // tier2 kan vere sant medan noCanvas trekkjer lerretet attende – då skal
  // tavla oppføre seg nøyaktig som om tier 2 ikkje fanst, både for lerretet
  // sjølv og for CSS-uskarpleiken det elles ville ha overteke for.
  const tier2Active = tier2 && !noCanvas;

  // Per-teikn-utsnitt lagar vi berre når vi faktisk måler per teikn.
  // Starteksten får ingen uskarpleik: han vart gjeven, ikkje gjetta.
  const spans = useMemo(() => {
    if (!gauge || gauge.kind !== "conf" || !text) return null;
    return Array.from(text).map((ch, i) => {
      const j = i - gauge.promptLen;
      if (j < 0) return { ch, smudge: 0 };
      // conf[j] kan mangle om teksten er kutta midt i skrivinga
      const p = gauge.conf[j];
      return { ch, smudge: p === undefined ? 0 : confToSmudge(p) };
    });
  }, [gauge, text]);

  // Tap-måleren gjeld heile linja under eitt – tapet *er* ein global skalar.
  // Utan tekst er det ingenting å måle: plassholdaren skal aldri visast uklar.
  const lineSmudge =
    gauge?.kind === "loss" && text
      ? 1 - lossToFocus(gauge.value, gauge.vocab)
      : 0;

  // Uklarleikskartet: eit lite lerret med same geometri som tavla, der kvitt
  // = heilt uklart. For per-teikn-måling teiknar vi éin rute per teikn-utsnitt
  // ut frå den faktiske plasseringa, så det held sjølv når linja bryt.
  // For tap-måling er heile flata éin verdi.
  const buildSmudgeMap = useCallback(() => {
    const el = boardRef.current;
    if (!el || !gauge) return null;
    const r = el.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width / 4));
    const h = Math.max(1, Math.round(r.height / 4));
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) return null;

    if (gauge.kind === "loss") {
      const v = Math.round(lineSmudge * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(0, 0, w, h);
      return c;
    }

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);
    // berre teikn-utsnitta, aldri plassholdaren – elles forskyv indeksane seg
    const kids = el.querySelectorAll("span[data-ch]");
    const list = spans ?? [];
    kids.forEach((node, i) => {
      const s = list[i]?.smudge ?? 0;
      if (s <= 0.02) return;
      const b = node.getBoundingClientRect();
      const v = Math.round(s * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect((b.x - r.x) / 4, (b.y - r.y) / 4, b.width / 4, b.height / 4);
    });
    return c;
  }, [gauge, lineSmudge, spans]);

  useLayoutEffect(() => {
    if (!tier2Active || !gauge) return;
    const el = boardRef.current;
    const canvas = canvasRef.current;
    if (!el || !canvas) return;

    const gl = canvas.getContext("webgl2", { premultipliedAlpha: true });
    if (!gl) {
      setTier2(false); // ingen WebGL2 likevel: fall til CSS-nivået
      return;
    }

    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(sh) ?? "shader compile failed");
      }
      return sh;
    };

    let prog: WebGLProgram;
    try {
      prog = gl.createProgram()!;
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(prog) ?? "link failed");
      }
    } catch {
      setTier2(false);
      return;
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW
    );
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const mkTex = (unit: number) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      return t;
    };
    const boardTex = mkTex(0);
    const smudgeTex = mkTex(1);
    gl.uniform1i(gl.getUniformLocation(prog, "u_board"), 0);
    gl.uniform1i(gl.getUniformLocation(prog, "u_smudge"), 1);
    const uTexel = gl.getUniformLocation(prog, "u_texel");
    const uTime = gl.getUniformLocation(prog, "u_time");

    // Rørsle er pynt; uklarleik er informasjon. Ved redusert rørsle frys tida,
    // men kartet blir teikna som før.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;
    let alive = true;
    const t0 = performance.now();

    const frame = () => {
      if (!alive) return;
      const r = el.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(r.width * dpr));
      const h = Math.max(1, Math.round(r.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.uniform2f(uTexel, 1 / w, 1 / h);
      gl.uniform1f(uTime, still ? 0 : (performance.now() - t0) / 1000);

      const map = buildSmudgeMap();
      if (map) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, smudgeTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, map);
      }

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, boardTex);
      // den eksperimentelle utvidinga: levande DOM rett inn som tekstur
      (gl as unknown as {
        texElementImage2D: (
          target: number, level: number, internalformat: number,
          format: number, type: number, element: Element
        ) => void;
      }).texElementImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      raf = requestAnimationFrame(frame);
    };

    try {
      frame();
    } catch {
      alive = false;
      setTier2(false); // utvidinga finst ikkje likevel
      return;
    }

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      gl.deleteProgram(prog);
      gl.deleteBuffer(buf);
      gl.deleteTexture(boardTex);
      gl.deleteTexture(smudgeTex);
    };
  }, [tier2Active, gauge, buildSmudgeMap]);

  return (
    <div className={cn("space-y-2", className)}>
      <div className="tavle p-4">
        <div className="mb-2 flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-kritt/70">
          {children}
          {label}
        </div>
        {/* Innpakninga er posisjonert slik at lerretet i oppgåve 5 kan leggje
            seg nøyaktig oppå teksten – ikkje oppå etiketten. */}
        <div className="relative">
          <p
            ref={boardRef}
            className={textClassName}
            style={
              // Berre stil linja når det faktisk finst modell-tekst å vise fram –
              // elles ville plassholdaren (vanleg UI, ikkje modell-utdata) blitt
              // uklar saman med han. Sjekket her, ikkje berre ved kallestaden,
              // held gjeld for alle framtidige brukarar av <Tavle>. Når tier 2
              // er aktiv gjer shaderen denne jobben i staden, så CSS-filteret
              // vert undertrykt her.
              !tier2Active && gauge?.kind === "loss" && text
                ? {
                    filter: `blur(${blurPx(lineSmudge).toFixed(2)}px)`,
                    opacity: chalkOpacity(lineSmudge),
                  }
                : undefined
            }
          >
            {!text && placeholder && <span className="text-kritt/50">{placeholder}</span>}
            {spans
              ? spans.map((s, i) => (
                  // data-ch merkjer teikn-utsnitta, slik at uklarleikskartet i
                  // oppgåve 5 kan finne akkurat dei og ikkje t.d. plassholdaren
                  <span
                    key={i}
                    data-ch=""
                    style={
                      !tier2Active && s.smudge > 0.02
                        ? {
                            filter: `blur(${blurPx(s.smudge).toFixed(2)}px)`,
                            opacity: chalkOpacity(s.smudge),
                          }
                        : undefined
                    }
                  >
                    {s.ch}
                  </span>
                ))
              : text}
          </p>
          {tier2Active && gauge && (
            // Lerretet ligg oppå tavla og skjuler henne. Teksten under er
            // urørt og fullt levande: markering, kopiering, Cmd+F og
            // skjermlesarar går rett gjennom (pointer-events: none).
            <canvas
              ref={canvasRef}
              /* @ts-expect-error layoutsubtree er frå html-in-canvas-forsøket */
              layoutsubtree=""
              aria-hidden
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
          )}
        </div>
      </div>
      {/* Måleren er reint visuell. Samandraget ber same talet i ord, slik at
          skjermlesarar – og folk som berre vil ha talet – får det same.
          Utan tekst er det ingen måling å melde frå om, så samandraget
          ligg nede saman med sjølve uklarleiken. */}
      {gauge && text && (
        <p className="text-xs leading-relaxed text-blyant">
          {summary} {legend}
        </p>
      )}
    </div>
  );
}
