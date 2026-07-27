import { useEffect, useRef } from "react";

// Spør før noko som kastar bort trening. Alle slike val i appen går gjennom
// denne eine ruta, så «kva skjer no?» ser likt ut kvar gong.

export interface Ask {
  /** Overskrift: kva som kjem til å skje. */
  title: string;
  /** Handlinga brukaren nettopp gjorde. */
  what: string;
  /** Kor mykje arbeid som står på spel, om det finst noko. */
  note?: string;
  /** Følgja, med vanlege ord. */
  body: string;
  yes: string;
  no: string;
  onYes: () => void;
}

export default function Bekreft({ ask, onClose }: { ask: Ask | null; onClose: () => void }) {
  const yesRef = useRef<HTMLButtonElement | null>(null);
  const open = ask !== null;

  // Escape avbryt, og knappen «ja» får fokus – då kan heile ruta styrast frå
  // tastaturet, og Tab held seg mellom dei to knappane.
  useEffect(() => {
    if (!open) return;
    yesRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!ask) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-blekk/40 p-4"
      onClick={onClose}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="bekreft-tittel"
        aria-describedby="bekreft-tekst"
        className="panel w-full max-w-md bg-papir p-5 sm:p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="bekreft-tittel" className="text-lg font-bold text-blekk">
          {ask.title}
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-blekk">{ask.what}</p>
        {ask.note && (
          <p className="mt-2 font-mono text-xs font-semibold text-rettepenn">{ask.note}</p>
        )}
        <p id="bekreft-tekst" className="mt-2 text-sm leading-relaxed text-blyant">
          {ask.body}
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <button
            ref={yesRef}
            type="button"
            onClick={() => {
              ask.onYes();
              onClose();
            }}
            className="knapp knapp-rettepenn"
          >
            {ask.yes}
          </button>
          <button type="button" onClick={onClose} className="knapp knapp-omriss">
            {ask.no}
          </button>
        </div>
      </div>
    </div>
  );
}
