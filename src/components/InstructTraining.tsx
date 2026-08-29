import { Gloss } from "@/components/Gloss";
import { Card } from "@/components/ui";
import type { Strings } from "@/lib/i18n";

export default function InstructTraining({ s }: { s: Strings["instruct"] }) {
  return (
    <div className="space-y-5">
      <Card className="space-y-6">
        <div>
          <div className="etikett text-rettepenn">{s.kicker}</div>
          <h3 className="mt-1 text-xl font-bold tracking-tight text-blekk">{s.title}</h3>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-blyant">
            <Gloss text={s.intro} />
          </p>
        </div>

        <div className="grid grid-cols-1 items-stretch gap-3 sm:grid-cols-[1fr_auto_1.35fr]">
          <div className="rounded-[3px] border border-blekk/30 bg-papir/70 p-4">
            <div className="etikett text-[10px]">{s.beforeLabel}</div>
            <p className="mt-3 font-mono text-sm text-blekk">{s.beforeExample}</p>
            <p className="mt-2 text-xs leading-relaxed text-blyant">{s.beforeHelp}</p>
          </div>

          <div aria-hidden className="flex items-center justify-center font-display text-xl text-rettepenn">
            <span className="sm:hidden">↓</span>
            <span className="hidden sm:inline">→</span>
          </div>

          <div className="overflow-hidden rounded-[3px] border-2 border-blekk font-mono text-xs">
            <div className="etikett bg-blekk px-3 py-2 text-[10px] text-white">{s.afterLabel}</div>
            <div className="grid grid-cols-[6.5rem_1fr] border-b border-blekk/25 bg-papir/70">
              <div className="etikett border-r border-blekk/25 px-3 py-3 text-[10px]">{s.promptLabel}</div>
              <div className="px-3 py-3 leading-relaxed text-blekk">{s.prompt}</div>
            </div>
            <div className="grid grid-cols-[6.5rem_1fr] bg-tusj/55">
              <div className="etikett border-r border-blekk/25 px-3 py-3 text-[10px]">{s.answerLabel}</div>
              <div className="px-3 py-3 leading-relaxed text-blekk">{s.answer}</div>
            </div>
          </div>
        </div>

        <p className="rounded-[2px] bg-papir/70 p-3 text-sm leading-relaxed text-blyant">
          <b className="text-blekk">{s.sameModelLead}</b>{" "}
          <Gloss text={s.sameModelBody} />
        </p>

        <div className="flex flex-wrap items-baseline justify-between gap-2 border-t border-blekk/20 pt-4 font-mono text-[11px] leading-relaxed text-blyant">
          <span>{s.paperCredit}</span>
          <a href="#lesmer" className="font-semibold text-blekk underline decoration-dotted underline-offset-4">
            {s.paperLink}
          </a>
        </div>
      </Card>

      <p className="text-xs leading-relaxed text-blyant">
        <b className="text-blekk">{s.scopeTitle}</b>{" "}
        <Gloss text={s.scopeBody} />
      </p>

      <h3 className="pt-1 text-lg font-bold tracking-tight text-blekk">{s.handoffTitle}</h3>
    </div>
  );
}
