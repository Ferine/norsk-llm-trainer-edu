import type { Strings } from "@/lib/i18n";
import type { MoeConfig } from "@/lib/ml";
import { Gloss } from "@/components/Gloss";

interface Props {
  layers: number;
  heads: number;
  dim: number;
  moe?: MoeConfig;
  s: Strings;
}

// Tre toner med mening: papir = enkeltsteg, rute = de gjentatte blokkene,
// tusj = svaret modellen gir fra seg (markert som med overstrykningstusj).
function Box({
  title,
  sub,
  tone = "papir",
  children,
}: {
  title: string;
  sub?: string;
  tone?: "papir" | "rute" | "tusj";
  children?: React.ReactNode;
}) {
  const tones: Record<string, string> = {
    papir: "bg-white",
    rute: "bg-rute/25",
    tusj: "bg-tusj/40",
  };
  return (
    <div className={`rounded-[3px] border-2 border-blekk px-3 py-2 text-center text-blekk ${tones[tone]}`}>
      <div className="text-xs font-semibold sm:text-sm"><Gloss text={title} /></div>
      {sub && <div className="font-mono text-[10px] text-blyant sm:text-[11px]"><Gloss text={sub} /></div>}
      {children}
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center py-1 text-blyant/60">
      <svg width="16" height="18" viewBox="0 0 16 18" fill="none">
        <path d="M8 0v14M3 9l5 5 5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

// Skjematisk teikning av transformer-arkitekturen (GPT-stil).
export default function Architecture({ layers, heads, dim, moe, s }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto_1fr]">
      {/* Hovudstraum (venstre/sentralt) */}
      <div className="space-y-0 md:order-2">
        <Box title={s.arch.boxInput.title} sub={s.arch.boxInput.sub} />
        <Arrow />
        <Box title={s.arch.boxEmbedding.title} sub={s.arch.boxEmbedding.sub(dim)} />
        <Arrow />
        {Array.from({ length: layers }).map((_, i) => (
          <div key={i}>
            <Box title={s.arch.boxBlock.title(i + 1)} sub={s.arch.boxBlock.sub} tone="rute">
              <div className="mt-2 space-y-1">
                <Box title={s.arch.boxAttn.title} sub={s.arch.boxAttn.sub(heads)} />
                {moe ? (
                  <Box
                    title={s.arch.boxMoe.title}
                    sub={s.arch.boxMoe.sub(moe.experts, moe.topK)}
                  />
                ) : (
                  <Box title={s.arch.boxFfn.title} sub={s.arch.boxFfn.sub} />
                )}
                <div className="font-mono text-[10px] text-blyant"><Gloss text={s.arch.residualNote} /></div>
              </div>
            </Box>
            <Arrow />
          </div>
        ))}
        <Box title={s.arch.boxFinalNorm.title} sub={s.arch.boxFinalNorm.sub} />
        <Arrow />
        <Box title={s.arch.boxOutHead.title} sub={s.arch.boxOutHead.sub} tone="tusj" />
        <Arrow />
        <Box title={s.arch.boxSoftmax.title} sub={s.arch.boxSoftmax.sub} tone="tusj" />
      </div>

      {/* Forklaring (høgre på stor skjerm) */}
      <aside className="md:order-3">
        <div className="rounded-[3px] border-2 border-blekk bg-white p-4 text-sm leading-relaxed text-blyant">
          <h4 className="mb-2 font-semibold text-blekk">{s.arch.explainHeading}</h4>
          <ul className="space-y-2">
            {s.arch.explain.map((e, i) => (
              <li key={i}>
                <b className="text-blekk"><Gloss text={e.b} /></b> <Gloss text={e.t} />
              </li>
            ))}
          </ul>
        </div>
      </aside>

      {/* plasshaldar for grid på stor skjerm */}
      <div className="hidden md:order-1 md:block" />
    </div>
  );
}
