import { useMemo } from "react";
import type { ReactNode } from "react";
import { cn } from "@/utils/cn";
import { blurPx, chalkOpacity, confToSmudge, lossToFocus } from "@/lib/chalk";

// Tavla der modellen skriv. Teksten kan teiknast med «krit som ikkje har
// sett seg»: kor uklart eit teikn står, er bunde til eit ekte tal frå
// modellen – anten tapet (heile linja) eller sikkerheita per teikn.
// Utan `gauge` oppfører komponenten seg nøyaktig som tavla gjorde før.

export type Gauge =
  | { kind: "loss"; value: number; vocab: number }
  | { kind: "conf"; conf: Float32Array; promptLen: number };

interface Props {
  label: string;
  text: string;
  placeholder: string;
  legend: string;
  summary: string;
  gauge?: Gauge;
  className?: string;
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
  children,
}: Props) {
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
  const lineSmudge =
    gauge?.kind === "loss" ? 1 - lossToFocus(gauge.value, gauge.vocab) : 0;

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
            className="min-h-8 whitespace-pre-wrap font-mono text-sm leading-relaxed text-kritt"
            style={
              gauge?.kind === "loss"
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
                      s.smudge > 0.02
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
        </div>
      </div>
      {/* Måleren er reint visuell. Samandraget ber same talet i ord, slik at
          skjermlesarar – og folk som berre vil ha talet – får det same. */}
      {gauge && (
        <p className="text-xs leading-relaxed text-blyant">
          {summary} {legend}
        </p>
      )}
    </div>
  );
}
