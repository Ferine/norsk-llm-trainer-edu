import type { Lang } from "./i18n";

/* --- Ordlista -----------------------------------------------------------
   Fagorda i appen, med ei kjapp gloseforklaring på begge målformer.
   Gloss-komponenten (components/Gloss.tsx) leitar opp orda i løpande tekst
   og heng forklaringa på som ein handskriven lapp når ein peikar på ordet.

   `ord` er skrivemåtane som skal merkjast, med bøygde former der teksten
   brukar dei («tapet», «merksemda»). glossify sorterer sjølv lengste først,
   så «tap (loss)» vinn over «tap». Berre første førekomst i kvar tekstbolk
   blir merkt – elles blir sida sjåande ut som eit understrekingsseminar. */

export interface Fagord {
  ord: string[];
  def: string;
}

export type OrdId =
  | "spraakmodell"
  | "token"
  | "transformer"
  | "tap"
  | "vekter"
  | "backpropagation"
  | "embedding"
  | "attention"
  | "kausal-maskering"
  | "feed-forward"
  | "residual"
  | "layernorm"
  | "logits"
  | "softmax"
  | "korpus"
  | "vokabular"
  | "minibatch"
  | "laeringsrate"
  | "adam"
  | "muon"
  | "gelu"
  | "situ-glu"
  | "moe"
  | "rlhf"
  | "dpo"
  | "temperatur"
  | "top-k"
  | "gguf"
  | "autograd"
  | "nevralt-nett"
  | "parametere";

/* Gløymer du ei målform på eit ord, kompilerer det ikkje. */
export const ORDLISTE: Record<OrdId, Record<Lang, Fagord>> = {
  spraakmodell: {
    bm: {
      ord: ["språkmodell", "språkmodellen", "språkmodeller", "språkmodellene"],
      def: "Et dataprogram trent til én ting: å gjette neste tegn eller ord i en tekst. All skrivingen er bare gjetting, om og om igjen.",
    },
    nn: {
      ord: ["språkmodell", "språkmodellen", "språkmodellar", "språkmodellane"],
      def: "Eit dataprogram trena til éin ting: å gjette neste teikn eller ord i ein tekst. All skrivinga er berre gjetting, om att og om att.",
    },
  },
  token: {
    bm: {
      ord: ["token", "tokenisering"],
      def: "Den minste tekstbiten modellen regner med – her ett enkelt tegn, i store modeller en ord-bit. Hvert token får sitt eget nummer.",
    },
    nn: {
      ord: ["token", "tokenisering"],
      def: "Den minste tekstbiten modellen reknar med – her eitt enkelt teikn, i store modellar ein ord-bit. Kvart token får sitt eige nummer.",
    },
  },
  transformer: {
    bm: {
      ord: ["transformer", "transformeren"],
      def: "Byggetegningen bak moderne språkmodeller (fra 2017): lag på lag som først ser på sammenhengen i teksten, så tenker videre på den.",
    },
    nn: {
      ord: ["transformer", "transformeren"],
      def: "Byggjeteikninga bak moderne språkmodellar (frå 2017): lag på lag som først ser på samanhengen i teksten, så tenkjer vidare på han.",
    },
  },
  tap: {
    bm: {
      ord: ["tap (loss)", "tap", "tapet"],
      def: "Feilmålet i treningen: hvor overrasket modellen ble av det riktige neste tegnet. Lavere tap = bedre gjetting.",
    },
    nn: {
      ord: ["tap (loss)", "tap", "tapet"],
      def: "Feilmålet i treninga: kor overraska modellen vart av det rette neste teiknet. Lågare tap = betre gjetting.",
    },
  },
  vekter: {
    bm: {
      ord: ["vektene", "vekter"],
      def: "Tallene inni modellen som lagrer alt den har lært. Å trene = å justere disse tallene bitte litt, tusenvis av ganger.",
    },
    nn: {
      ord: ["vektene", "vektane", "vekter"],
      def: "Tala inni modellen som lagrar alt han har lært. Å trene = å justere desse tala bitte litt, tusenvis av gonger.",
    },
  },
  backpropagation: {
    bm: {
      ord: ["backpropagation"],
      def: "Regnemetoden som sporer feilen bakover gjennom alle lagene, så hver vekt får vite hvilken vei den skal vris. På norsk: tilbakeføring av feil.",
    },
    nn: {
      ord: ["backpropagation"],
      def: "Reknemetoden som sporar feilen bakover gjennom alle laga, så kvar vekt får vite kva veg ho skal vridast. På norsk: tilbakeføring av feil.",
    },
  },
  embedding: {
    bm: {
      ord: ["embedding", "innbygging", "innbyggingen"],
      def: "Oversettelsen fra tegn til en liste med tall (en vektor) som modellen kan regne med. Tegn som ligner, får liknende tall.",
    },
    nn: {
      ord: ["embedding", "innbygging", "innbygginga"],
      def: "Omsetjinga frå teikn til ei liste med tal (ein vektor) som modellen kan rekne med. Teikn som liknar, får liknande tal.",
    },
  },
  attention: {
    bm: {
      ord: ["multi-head oppmerksomhet", "selvoppmerksomhet", "oppmerksomhet", "oppmerksomheten"],
      def: "Mekanismen (attention) der hvert tegn ser tilbake på teksten før seg og plukker ut det som er viktig akkurat nå. Hjertet i transformeren.",
    },
    nn: {
      ord: ["multi-head merksemd", "sjølvmerksemd", "merksemd", "merksemda"],
      def: "Mekanismen (attention) der kvart teikn ser attende på teksten før seg og plukkar ut det som er viktig akkurat no. Hjartet i transformeren.",
    },
  },
  "kausal-maskering": {
    bm: {
      ord: ["kausal maskering"],
      def: "Sperren som skjuler fremtiden: når modellen gjetter et tegn, ser den bare tegnene som kom før – aldri fasiten.",
    },
    nn: {
      ord: ["kausal maskering"],
      def: "Sperra som gøymer framtida: når modellen gjettar eit teikn, ser han berre teikna som kom før – aldri fasiten.",
    },
  },
  "feed-forward": {
    bm: {
      ord: ["feed-forward"],
      def: "Et lite nevralt nett inni hver blokk som bearbeider hvert tegn for seg, etter at oppmerksomheten har hentet inn sammenhengen.",
    },
    nn: {
      ord: ["feed-forward"],
      def: "Eit lite nevralt nett inni kvar blokk som arbeider vidare med kvart teikn for seg, etter at merksemda har henta inn samanhengen.",
    },
  },
  residual: {
    bm: {
      ord: ["residualveier", "residual"],
      def: "En snarvei der signalet hopper forbi et ledd og legges til igjen etterpå. Da kan ingen blokk ødelegge det som allerede er lært.",
    },
    nn: {
      ord: ["residualvegar", "residual"],
      def: "Ein snarveg der signalet hoppar forbi eit ledd og blir lagt til att etterpå. Då kan inga blokk øydeleggje det som alt er lært.",
    },
  },
  layernorm: {
    bm: {
      ord: ["LayerNorm"],
      def: "En liten opprydding som skalerer tallene til passe størrelse mellom leddene, så treningen holder seg stabil.",
    },
    nn: {
      ord: ["LayerNorm"],
      def: "Ei lita opprydding som skalerer tala til passe storleik mellom ledda, så treninga held seg stabil.",
    },
  },
  logits: {
    bm: {
      ord: ["logits"],
      def: "Rå poengsum for hvert mulige neste tegn – før softmax gjør dem om til sannsynlighet.",
    },
    nn: {
      ord: ["logits"],
      def: "Rå poengsum for kvart moglege neste teikn – før softmax gjer dei om til sannsyn.",
    },
  },
  softmax: {
    bm: {
      ord: ["softmax"],
      def: "Regnestykket som gjør poengsummene om til sannsynligheter som til sammen blir 100 %. Størst poeng → størst sjanse.",
    },
    nn: {
      ord: ["softmax"],
      def: "Reknestykket som gjer poengsummane om til sannsyn som til saman blir 100 %. Størst poeng → størst sjanse.",
    },
  },
  korpus: {
    bm: {
      ord: ["korpus", "korpuset"],
      def: "Tekstsamlingen modellen lærer av. Modellen blir som korpuset sitt – derfor betyr kvaliteten alt.",
    },
    nn: {
      ord: ["korpus", "korpuset"],
      def: "Tekstsamlinga modellen lærer av. Modellen blir som korpuset sitt – difor tyder kvaliteten alt.",
    },
  },
  vokabular: {
    bm: {
      ord: ["vokabular", "vokabularet"],
      def: "Lista over alle token modellen kjenner – her: hele tegnsettet. Alt modellen noensinne kan skrive, står her.",
    },
    nn: {
      ord: ["vokabular", "vokabularet"],
      def: "Lista over alle token modellen kjenner – her: heile teiknsettet. Alt modellen nokosinne kan skrive, står her.",
    },
  },
  minibatch: {
    bm: {
      ord: ["minibatch"],
      def: "En liten bunke tekstbiter modellen øver på i samme steg. Snittet av flere feil gir en jevnere vridning enn én bit alene.",
    },
    nn: {
      ord: ["minibatch"],
      def: "Ein liten bunke tekstbitar modellen øver på i same steg. Snittet av fleire feil gir ei jamnare vriding enn éin bit åleine.",
    },
  },
  laeringsrate: {
    bm: {
      ord: ["læringsrate", "læringsraten"],
      def: "Hvor mye skruene vris per steg. For mye: læringen kollapser i kaos. For lite: det tar evigheter.",
    },
    nn: {
      ord: ["læringsrate", "læringsraten"],
      def: "Kor mykje skruane blir vridne per steg. For mykje: læringa kollapsar i kaos. For lite: det tek all verdas tid.",
    },
  },
  adam: {
    bm: {
      ord: ["Adam"],
      def: "Arbeidshesten blant optimalisatorer siden 2015: gir hver enkelt vekt sin egen, selvjusterende skrittlengde.",
    },
    nn: {
      ord: ["Adam"],
      def: "Arbeidshesten blant optimalisatorar sidan 2015: gir kvar einskild vekt si eiga, sjølvjusterande skrittlengd.",
    },
  },
  muon: {
    bm: {
      ord: ["Muon"],
      def: "En ny optimalisator (2024) som vrir en hel tallmatrise under ett og jevner ut retningene. Brukes på de største modellene i dag.",
    },
    nn: {
      ord: ["Muon"],
      def: "Ein ny optimalisator (2024) som vrir ei heil talmatrise under eitt og jamnar ut retningane. Blir brukt på dei største modellane i dag.",
    },
  },
  gelu: {
    bm: {
      ord: ["GELU"],
      def: "En mye brukt aktiveringsfunksjon: en myk knekk som avgjør hvor mye av hvert signal som slipper videre i nettet.",
    },
    nn: {
      ord: ["GELU"],
      def: "Ein mykje brukt aktiveringsfunksjon: ein mjuk knekk som avgjer kor mykje av kvart signal som slepp vidare i nettet.",
    },
  },
  "situ-glu": {
    bm: {
      ord: ["SiTU-GLU"],
      def: "En portfunksjon fra Kimi K3-oppskriften: laget styrer selv hvor mye som slipper gjennom, og et innebygd tak hindrer tallene i å løpe løpsk.",
    },
    nn: {
      ord: ["SiTU-GLU"],
      def: "Ein portfunksjon frå Kimi K3-oppskrifta: laget styrer sjølv kor mykje som slepp gjennom, og eit innebygd tak hindrar tala i å renne løpsk.",
    },
  },
  moe: {
    bm: {
      ord: ["MoE"],
      def: "«Mixture of Experts»: det brede laget deles i flere smale eksperter, og bare noen få vekkes per tegn. Like mange vekter, mye mindre regning.",
    },
    nn: {
      ord: ["MoE"],
      def: "«Mixture of Experts»: det breie laget blir delt i fleire smale ekspertar, og berre nokre få blir vekte per teikn. Like mange vekter, mykje mindre rekning.",
    },
  },
  rlhf: {
    bm: {
      ord: ["RLHF"],
      def: "«Reinforcement Learning from Human Feedback»: mennesker velger de beste svarene, og modellen justeres mot det folk foretrekker. Slik lærte chatbotene folkeskikk.",
    },
    nn: {
      ord: ["RLHF"],
      def: "«Reinforcement Learning from Human Feedback»: menneske vel dei beste svara, og modellen blir justert mot det folk føretrekkjer. Slik lærte chatbotane folkeskikk.",
    },
  },
  dpo: {
    bm: {
      ord: ["DPO"],
      def: "«Direct Preference Optimization»: en enkel RLHF-oppskrift som flytter modellen rett mot svarene du valgte – uten et eget nettverk som setter poeng på svarene.",
    },
    nn: {
      ord: ["DPO"],
      def: "«Direct Preference Optimization»: ei enkel RLHF-oppskrift som flyttar modellen rett mot svara du valde – utan eit eige nettverk som set poeng på svara.",
    },
  },
  temperatur: {
    bm: {
      ord: ["temperatur", "temperaturen"],
      def: "Styrer sjansespillet når neste tegn velges: 0 = alltid det sikreste tegnet, høyere = jevnere lodd og villere tekst.",
    },
    nn: {
      ord: ["temperatur", "temperaturen"],
      def: "Styrer sjansespelet når neste teikn blir valt: 0 = alltid det sikraste teiknet, høgare = jamnare lodd og villare tekst.",
    },
  },
  "top-k": {
    bm: {
      ord: ["top-k"],
      def: "Modellen får bare trekke blant de k mest sannsynlige tegnene – resten kastes før loddtrekningen.",
    },
    nn: {
      ord: ["top-k"],
      def: "Modellen får berre trekkje mellom dei k mest sannsynlege teikna – resten blir kasta før loddtrekkinga.",
    },
  },
  gguf: {
    bm: {
      ord: ["GGUF"],
      def: "Filformatet åpne modeller deles i (kjent fra llama.cpp): én fil med alle vektene pluss en liten innholdsfortegnelse.",
    },
    nn: {
      ord: ["GGUF"],
      def: "Filformatet opne modellar blir delte i (kjent frå llama.cpp): éi fil med alle vektene pluss ei lita innhaldsliste.",
    },
  },
  autograd: {
    bm: {
      ord: ["autograd"],
      def: "Maskineriet som automatisk regner ut hvilken vei hver vekt må vris – det som gjør backpropagation mulig å skrive.",
    },
    nn: {
      ord: ["autograd"],
      def: "Maskineriet som automatisk reknar ut kva veg kvar vekt må vridast – det som gjer backpropagation mogleg å skrive.",
    },
  },
  "nevralt-nett": {
    bm: {
      ord: ["nevralt nett", "nevrale nett"],
      def: "Et nettverk av enkle regneenheter med justerbare vekter imellom – grunnmuren i moderne maskinlæring.",
    },
    nn: {
      ord: ["nevralt nett", "nevrale nett"],
      def: "Eit nettverk av enkle rekneeiningar med justerbare vekter imellom – grunnmuren i moderne maskinlæring.",
    },
  },
  parametere: {
    bm: {
      ord: ["parametere", "parametre"],
      def: "Samlenavn på alle tallene modellen lærer (vektene). Flere parametere = større modell – og mer regning.",
    },
    nn: {
      ord: ["parametrar", "parametrane"],
      def: "Samlenamn på alle tala modellen lærer (vektene). Fleire parametrar = større modell – og meir rekning.",
    },
  },
};

/* --- Oppslag i løpande tekst ------------------------------------------- */

export interface OrdTreff {
  id: OrdId;
  /** Ordet slik det står i teksten (med original stor/liten bokstav). */
  ord: string;
  def: string;
}

export type GlossDel = string | OrdTreff;

const cache = new Map<Lang, { re: RegExp; byOrd: Map<string, OrdId> }>();

function matcher(lang: Lang) {
  let m = cache.get(lang);
  if (!m) {
    const byOrd = new Map<string, OrdId>();
    for (const id of Object.keys(ORDLISTE) as OrdId[]) {
      for (const o of ORDLISTE[id][lang].ord) byOrd.set(o.toLowerCase(), id);
    }
    /* Lengste skrivemåte først, så «tap (loss)» vinn over «tap». Krav om
       ikkje-bokstav på begge sider gjer at «taper» og «tapetsering» går fri –
       men «GGUF-fila» treffer, sidan bindestrek ikkje er bokstav. */
    const alts = [...byOrd.keys()]
      .sort((a, b) => b.length - a.length)
      .map((o) => o.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re = new RegExp(`(?<![\\p{L}\\p{N}])(?:${alts.join("|")})(?![\\p{L}\\p{N}])`, "giu");
    m = { re, byOrd };
    cache.set(lang, m);
  }
  return m;
}

/** Deler ein tekst i vanlege strengar og fagord-treff. Berre første
 *  førekomst av kvart ord i teksten blir eit treff. */
export function glossify(text: string, lang: Lang): GlossDel[] {
  const { re, byOrd } = matcher(lang);
  const deler: GlossDel[] = [];
  const brukte = new Set<OrdId>();
  let sist = 0;
  re.lastIndex = 0;
  for (let m = re.exec(text); m !== null; m = re.exec(text)) {
    const id = byOrd.get(m[0].toLowerCase());
    if (id === undefined || brukte.has(id)) continue;
    brukte.add(id);
    if (m.index > sist) deler.push(text.slice(sist, m.index));
    deler.push({ id, ord: m[0], def: ORDLISTE[id][lang].def });
    sist = m.index + m[0].length;
  }
  if (sist < text.length) deler.push(text.slice(sist));
  return deler;
}
