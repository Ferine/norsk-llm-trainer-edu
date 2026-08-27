import { Gloss } from "@/components/Gloss";
import { sliceContextWindow } from "@/lib/context-window";
import type { Strings } from "@/lib/i18n";

function printable(ch: string): string {
  if (ch === " ") return "␣";
  if (ch === "\n") return "↵";
  if (ch === "\t") return "⇥";
  return ch;
}

function Character({ ch, forgotten = false }: { ch: string; forgotten?: boolean }) {
  return (
    <span
      className={
        forgotten
          ? "inline-flex h-7 min-w-6 items-center justify-center rounded-[2px] border border-dashed border-blyant/40 bg-white px-1 font-mono text-xs text-blyant/45 line-through"
          : "inline-flex h-7 min-w-6 items-center justify-center rounded-[2px] border border-blekk/35 bg-white px-1 font-mono text-xs font-semibold text-blekk"
      }
    >
      {printable(ch)}
    </span>
  );
}

export default function ContextWindow({
  tokens,
  capacity,
  s,
}: {
  tokens: readonly string[];
  capacity: number;
  s: Strings["chat"]["context"];
}) {
  const view = sliceContextWindow(tokens, capacity);

  return (
    <figure className="rounded-[3px] border-2 border-blekk bg-papir/70 p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <figcaption className="text-lg font-semibold text-blekk">{s.title}</figcaption>
        <span className="etikett">{s.used(view.used, capacity)}</span>
      </div>
      <p className="mt-1 max-w-2xl text-sm leading-relaxed text-blyant">
        <Gloss text={s.help(capacity)} />
      </p>

      {view.total === 0 ? (
        <p className="mt-4 rounded-[2px] border border-dashed border-blyant/40 bg-white p-3 text-sm text-blyant">
          {s.empty}
        </p>
      ) : (
        <>
          <p className="sr-only">{s.screenReader(view.forgotten, view.visible.join(""))}</p>
          <div aria-hidden className="mt-4 flex flex-wrap items-end gap-3">
            {view.forgotten > 0 && (
              <div>
                <div className="etikett mb-1">{s.outside(view.forgotten)}</div>
                <div className="flex flex-wrap gap-1 opacity-70">
                  {view.forgotten > view.forgottenPreview.length && (
                    <span className="inline-flex h-7 items-center font-mono text-sm text-blyant/50">…</span>
                  )}
                  {view.forgottenPreview.map((ch, i) => (
                    <Character key={`${view.forgotten - view.forgottenPreview.length + i}-${ch}`} ch={ch} forgotten />
                  ))}
                </div>
              </div>
            )}

            {view.forgotten > 0 && (
              <span className="pb-1 font-mono text-lg text-blyant/55">→</span>
            )}

            <div className="min-w-[12rem] flex-1">
              <div className="etikett mb-1">{s.inside}</div>
              <div className="flex min-h-11 flex-wrap gap-1 rounded-[3px] border-2 border-blekk bg-tusj/35 p-2">
                {view.visible.map((ch, i) => (
                  <Character key={`${view.forgotten + i}-${ch}`} ch={ch} />
                ))}
              </div>
            </div>

            <div className="flex-none text-center">
              <div className="etikett mb-1">{s.next}</div>
              <div className="inline-flex h-11 w-11 items-center justify-center rounded-[3px] border-2 border-rettepenn bg-white font-display text-xl text-rettepenn">
                ?
              </div>
            </div>
          </div>
          <p className="mt-3 font-mono text-[11px] leading-relaxed text-blyant">
            <Gloss text={s.oneChar} />
          </p>
        </>
      )}
    </figure>
  );
}
