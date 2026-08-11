import { createContext, useContext, useId, useMemo } from "react";
import type { Lang } from "@/lib/i18n";
import { glossify } from "@/lib/ordliste";

// Målforma glosene skal slåast opp i. Éin provider øvst i App held heile
// treet i takt med språkveljaren, så Gloss kan brukast kvar som helst utan
// å tre `lang` gjennom alle mellomledda.
export const GlossLang = createContext<Lang>("bm");

// Eitt fagord med gloselapp. Lappen er eit vanleg element (ikkje title=""),
// så han verkar med tastatur (tab + fokus) og på mobil (trykk).
function Fagord({ ord, def }: { ord: string; def: string }) {
  const id = useId();
  return (
    <span className="fagord" tabIndex={0} aria-describedby={id}>
      {ord}
      <span id={id} role="tooltip" className="fagord-lapp">
        {def}
      </span>
    </span>
  );
}

// Løpande tekst der fagorda frå ordlista får prikka strek og gloselapp.
export function Gloss({ text }: { text: string }) {
  const lang = useContext(GlossLang);
  const deler = useMemo(() => glossify(text, lang), [text, lang]);
  return (
    <>
      {deler.map((d, i) =>
        typeof d === "string" ? d : <Fagord key={i} ord={d.ord} def={d.def} />
      )}
    </>
  );
}
