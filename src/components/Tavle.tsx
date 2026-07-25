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
  // Utan tekst er det ingenting å måle: plassholdaren skal aldri visast uklar.
  const lineSmudge =
    gauge?.kind === "loss" && text
      ? 1 - lossToFocus(gauge.value, gauge.vocab)
      : 0;

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
            className={textClassName}
            style={
              // Berre stil linja når det faktisk finst modell-tekst å vise fram –
              // elles ville plassholdaren (vanleg UI, ikkje modell-utdata) blitt
              // uklar saman med han. Sjekket her, ikkje berre ved kallestaden,
              // held gjeld for alle framtidige brukarar av <Tavle>.
              gauge?.kind === "loss" && text
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
