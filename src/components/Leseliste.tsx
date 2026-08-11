import { LESELISTE, type Strings } from "@/lib/i18n";

// Skulebiblioteket bakarst i kladdeboka: éi hylle per panel, eitt bokkort per
// rad. Dette er dei einaste lenkjene i appen som peikar ut på nettet – difor
// står den ærlege noten øvst i staden for å bli gøymd.
export default function Leseliste({ s }: { s: Strings["readMore"] }) {
  return (
    <div className="space-y-4">
      <p className="max-w-2xl font-mono text-[11px] leading-relaxed text-blyant">{s.note}</p>

      {LESELISTE.map((hylle) => (
        <section key={hylle.id} className="panel px-4 py-3 sm:px-5">
          <h3 className="etikett">{s.shelves[hylle.id]}</h3>
          <ul>
            {hylle.items.map((lenke, i) => (
              <li key={lenke.id} className={i > 0 ? "border-t border-blekk/15" : ""}>
                <a
                  href={lenke.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="lesekort"
                >
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                    <span className="lesetittel font-semibold">{lenke.name}</span>
                    <span aria-hidden className="font-mono text-xs text-blyant">
                      ↗
                    </span>
                    {lenke.start && (
                      <span className="etikett rounded-[2px] bg-tusj px-1.5 py-0.5 text-blekk">
                        {s.startHere}
                      </span>
                    )}
                    {/* føringslinja treng plass, så ho står berre på brei skjerm */}
                    <span aria-hidden className="leder hidden sm:block" />
                    <span className="etikett">
                      {s.kinds[lenke.kind]}
                      {lenke.meta ? ` · ${lenke.meta}` : ""} · {s.levels[lenke.level]}
                    </span>
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-blyant">{lenke.by}</div>
                  <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-blekk">
                    {s.items[lenke.id]}
                  </p>
                  <span className="sr-only">{s.newTab}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
