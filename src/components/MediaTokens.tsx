import { Gloss } from "@/components/Gloss";
import type { Strings } from "@/lib/i18n";

const IMAGE_COLORS = [
  "#b9d5e8", "#9fc5de", "#f5df9a", "#d6e3b5",
  "#83adca", "#f0c975", "#dca48f", "#b9d6b0",
  "#527da1", "#d9b16b", "#9c7868", "#7fa27b",
];
const WAVE_HEIGHTS = [10, 19, 32, 22, 13, 26, 38, 20, 9, 17, 29, 14];

function VectorTokens({ labels }: { labels: string[] }) {
  return (
    <div className="flex flex-wrap justify-center gap-1">
      {labels.map((label) => (
        <span key={label} className="brikke px-2 py-1 font-mono text-[10px] text-blekk">
          {label}
        </span>
      ))}
    </div>
  );
}

export default function MediaTokens({ s }: { s: Strings["data"]["media"] }) {
  return (
    <section className="rounded-[3px] border border-blekk/25 bg-papir/70 p-4 sm:p-5">
      <h3 className="font-semibold text-blekk">{s.title}</h3>
      <p className="mt-1 max-w-2xl text-sm leading-relaxed text-blyant">
        <Gloss text={s.intro} />
      </p>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-[2px] border border-blekk/25 bg-white p-4">
          <h4 className="font-semibold text-blekk">{s.imageTitle}</h4>
          <div aria-hidden className="my-4 flex min-h-20 items-center justify-center gap-3">
            <div className="grid h-20 w-24 flex-none grid-cols-4 overflow-hidden rounded-[2px] border-2 border-blekk">
              {IMAGE_COLORS.map((color, i) => (
                <span key={i} className="border border-white/60" style={{ backgroundColor: color }} />
              ))}
            </div>
            <span className="font-display text-lg text-rettepenn">→</span>
            <VectorTokens labels={["v1", "v2", "v3", "…"]} />
          </div>
          <div className="etikett text-[10px] text-rettepenn">{s.imageFlow}</div>
          <p className="mt-1 text-xs leading-relaxed text-blyant">
            <Gloss text={s.imageBody} />
          </p>
        </div>

        <div className="rounded-[2px] border border-blekk/25 bg-white p-4">
          <h4 className="font-semibold text-blekk">{s.audioTitle}</h4>
          <div aria-hidden className="my-4 flex min-h-20 items-center justify-center gap-3">
            <div className="flex h-20 w-24 flex-none items-center justify-center gap-0.5 rounded-[2px] border-2 border-blekk bg-papir px-2">
              {WAVE_HEIGHTS.map((height, i) => (
                <span
                  key={i}
                  className="w-1 rounded-full bg-blekk"
                  style={{ height: `${height}px` }}
                />
              ))}
            </div>
            <span className="font-display text-lg text-rettepenn">→</span>
            <VectorTokens labels={["v1", "v2", "v3", "…"]} />
          </div>
          <div className="etikett text-[10px] text-rettepenn">{s.audioFlow}</div>
          <p className="mt-1 text-xs leading-relaxed text-blyant">
            <Gloss text={s.audioBody} />
          </p>
        </div>
      </div>

      <p className="mt-3 text-xs leading-relaxed text-blyant">
        <b className="text-blekk">{s.noteLead}</b>{" "}
        <Gloss text={s.noteBody} />
      </p>
    </section>
  );
}
