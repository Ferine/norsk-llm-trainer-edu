import { useContext } from "react";
import { cn } from "@/utils/cn";
import { GlossLang } from "@/components/Gloss";
import { ordlisteTema } from "@/lib/ordliste";
import type { Strings } from "@/lib/i18n";

// Glosene bakarst i kladdeboka: same forklaringane som gloselappane, men samla
// og sorterte etter tema, så dei går an å slå opp i staden for å jakte på
// prikka strekar rundt om på sida. Eitt panel per tema, oppslagsord i venstre
// spalte som i ei ekte ordliste.
export default function Ordliste({ s }: { s: Strings["ordliste"] }) {
  const lang = useContext(GlossLang);
  return (
    <div className="space-y-4">
      {ordlisteTema(lang).map((gruppe) => (
        <section key={gruppe.tema} className="panel px-4 py-3 sm:px-5">
          <h3 className="etikett">{s.temaer[gruppe.tema]}</h3>
          <dl>
            {gruppe.oppslag.map((o, i) => (
              <div
                key={o.id}
                className={cn(
                  "py-2 sm:grid sm:grid-cols-[13rem_1fr] sm:gap-x-4",
                  i > 0 && "border-t border-blekk/15"
                )}
              >
                <dt className="font-mono text-[13px] font-semibold leading-relaxed text-blekk">
                  {o.vis}
                </dt>
                <dd className="text-sm leading-relaxed text-blyant">{o.def}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}
