import type { ReactNode } from "react";
import { cn } from "@/utils/cn";
import type { Strings } from "@/lib/i18n";

// Ei «oppgave» i kladdeboka: nummeret står ute i margen, venstre for margstreken.
// Med `fold` blir oppgava valfri fordjuping – samanfalda som standard, slik at
// hovudløypa (t.d. i eit 15-minuttsverkstad) held seg kort.
export function Section({
  id,
  step,
  title,
  intro,
  children,
  fold,
}: {
  id: string;
  step: number;
  title: string;
  intro?: string;
  children: ReactNode;
  fold?: Strings["fold"];
}) {
  if (fold) {
    return (
      <section id={id} className="relative scroll-mt-24">
        <span aria-hidden className="margtall">
          {step}.
        </span>
        <details className="group">
          <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h2 className="text-2xl font-bold tracking-tight text-blekk">{title}</h2>
              <span className="etikett rounded-[2px] border border-blyant/40 px-1.5 py-0.5">
                {fold.tag}
              </span>
              <span className="font-mono text-xs font-semibold text-blekk group-open:hidden">
                {fold.show}
              </span>
              <span className="hidden font-mono text-xs font-semibold text-blekk group-open:inline">
                {fold.hide}
              </span>
            </div>
            {intro && <p className="mt-2 max-w-2xl leading-relaxed text-blyant">{intro}</p>}
          </summary>
          <div className="mt-5">{children}</div>
        </details>
      </section>
    );
  }
  return (
    <section id={id} className="relative scroll-mt-24">
      <span aria-hidden className="margtall">
        {step}.
      </span>
      <h2 className="text-2xl font-bold tracking-tight text-blekk">{title}</h2>
      {intro && <p className="mt-2 max-w-2xl leading-relaxed text-blyant">{intro}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("panel p-5 sm:p-6", className)}>{children}</div>;
}

// Samanfalda blokk for kontrollar/tal folk flest ikkje treng sjå.
export function Advanced({
  label,
  className,
  children,
}: {
  label: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={cn("group", className)}>
      <summary className="etikett cursor-pointer list-none [&::-webkit-details-marker]:hidden">
        <span className="group-open:hidden">+ </span>
        <span className="hidden group-open:inline">− </span>
        {label}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}
