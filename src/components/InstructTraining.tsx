import { Gloss } from "@/components/Gloss";
import { Card } from "@/components/ui";
import type { Strings } from "@/lib/i18n";

export default function InstructTraining({
  s,
  fold,
}: {
  s: Strings["instruct"];
  fold: Strings["fold"];
}) {
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

      <details className="group rounded-[3px] border border-blekk/25 bg-white p-4 sm:p-5">
        <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <h4 className="font-semibold text-blekk">{s.toolUse.title}</h4>
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
        </summary>

        <div className="mt-3">
          <p className="max-w-2xl text-sm leading-relaxed text-blyant">
            <Gloss text={s.toolUse.intro} />
          </p>
          <ol className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
            {s.toolUse.steps.map((stage, i) => (
              <li
                key={stage.title}
                className={`rounded-[2px] p-3 ${
                  i === 1
                    ? "border-2 border-rettepenn bg-tusj/40"
                    : "border border-blekk/25 bg-papir/70"
                }`}
              >
                <div className={`etikett text-[10px] ${i === 1 ? "text-rettepenn" : ""}`}>
                  {i + 1} · {stage.title}
                </div>
                <p className="mt-1 text-xs leading-relaxed text-blyant">
                  <Gloss text={stage.body} />
                </p>
              </li>
            ))}
          </ol>
          <p className="mt-3 text-xs leading-relaxed text-blyant">
            <b className="text-blekk">{s.toolUse.runtimeLead}</b>{" "}
            <Gloss text={s.toolUse.runtimeBody} />
          </p>
          <p className="mt-3 rounded-[2px] bg-papir/70 p-3 text-xs leading-relaxed text-blyant">
            <b className="text-blekk">{s.toolUse.mcpLead}</b>{" "}
            <Gloss text={s.toolUse.mcpBody} />
          </p>
        </div>
      </details>

      <p className="text-xs leading-relaxed text-blyant">
        <b className="text-blekk">{s.scopeTitle}</b>{" "}
        <Gloss text={s.scopeBody} />
      </p>

      <h3 className="pt-1 text-lg font-bold tracking-tight text-blekk">{s.handoffTitle}</h3>
    </div>
  );
}
