interface Props {
  layers: number;
  heads: number;
  dim: number;
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
export default function Architecture({ layers, heads, dim }: Props) {
  return (
    <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr]">
      {/* Hovudstraum (venstre/sentralt) */}
      <div className="space-y-0 md:order-2">
        <Box title="Inndata" sub="tekst → teikn (token-id-ar)" tone="slate" />
        <Arrow />
        <Box title="Innbygging (embedding)" sub={`teikn + posisjon → ${dim} tal`} tone="indigo" />
        <Arrow />
        {Array.from({ length: layers }).map((_, i) => (
          <div key={i}>
            <Box title={`Transformer-blokk ${i + 1}`} sub="sjølvoppmerksomhet + feed-forward" tone="violet">
              <div className="mt-2 space-y-1">
                <Box title="LayerNorm → Multi-head oppmerksomheit" sub={`${heads} hovud`} tone="violet" />
                <Box title="LayerNorm → Feed-forward (GELU)" sub="ikkje-lineær tenking" tone="violet" />
                <div className="text-[10px] opacity-70">+ residual-vegar (sprang over ledd)</div>
              </div>
            </Box>
            <Arrow />
          </div>
        ))}
        <Box title="Slutt-normalisering" sub="LayerNorm" tone="emerald" />
        <Arrow />
        <Box title="Utgangshovud" sub="→ poengsum (logits) for kvart teikn" tone="amber" />
        <Arrow />
        <Box title="Softmax" sub="→ sannsyn for kva teikn som kjem neste" tone="amber" />
      </div>

      {/* Forklaring (høgre på stor skjerm) */}
      <aside className="md:order-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          <h4 className="mb-2 font-semibold text-slate-800">Kva skjer inni?</h4>
          <ul className="space-y-2">
            <li><b className="text-indigo-600">Innbygging:</b> kvart teikn blir til ei liste med tal, og vi legg til informasjon om <i>kvar</i> i teksten det står.</li>
            <li><b className="text-violet-600">Sjølvoppmerksomheit:</b> kvart teikn ser på dei andre teikna og finn ut kva som er viktig i samanhengen.</li>
            <li><b className="text-violet-600">Feed-forward:</b> eit lite nevralt nett som "tenkjer" vidare over kvar posisjon.</li>
            <li><b className="text-emerald-600">Residualvegar:</b> informasjonen hoppar over kvart ledd slik at ingenting går tapt.</li>
            <li><b className="text-amber-600">Softmax:</b> gjer poenga om til sannsyn – slik vel modellen neste teikn.</li>
          </ul>
        </div>
      </aside>

      {/* plasshaldar for grid på stor skjerm */}
      <div className="hidden md:order-1 md:block" />
    </div>
  );
}
