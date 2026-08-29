import type { Lang } from "./i18n";

/* --- Ordlista -----------------------------------------------------------
   Fagorda i appen, med ei kjapp gloseforklaring på begge målformer. Éi kjelde,
   to bruksmåtar:

   – Gloss-komponenten (components/Gloss.tsx) leitar opp orda i løpande tekst
     og heng forklaringa på som ein handskriven lapp når ein peikar på ordet.
   – Ordliste-komponenten (components/Ordliste.tsx, steg 11) viser alle orda
     samla, sorterte etter tema, som glosene bakarst i ei kladdebok.

   `ord` er skrivemåtane som skal merkjast, med bøygde former der teksten
   brukar dei («tapet», «merksemda»). glossify sorterer sjølv lengste først,
   så «tap (loss)» vinn over «tap». Berre første førekomst i kvar tekstbolk
   blir merkt – elles blir sida sjåande ut som eit understrekingsseminar.
   `vis` er oppslagsforma slik ho står i ordlista. */

export interface Fagord {
  /** Oppslagsforma i ordlista (steg 11). */
  vis: string;
  /** Skrivemåtane som skal merkjast i løpande tekst. */
  ord: string[];
  def: string;
}

/* Temaa følgjer løypa gjennom appen; rekkjefølgja her er rekkjefølgja på sida. */
export type Tema =
  | "grunnlag"
  | "tekst"
  | "inni"
  | "laering"
  | "optimalisering"
  | "tal"
  | "skriv"
  | "finpuss"
  | "skala";

export const TEMA_REKKEFOLGJE: Tema[] = [
  "grunnlag",
  "tekst",
  "inni",
  "laering",
  "optimalisering",
  "tal",
  "skriv",
  "finpuss",
  "skala",
];

export interface Oppslag {
  tema: Tema;
  bm: Fagord;
  nn: Fagord;
}

export type OrdId =
  // grunnlaget
  | "spraakmodell"
  | "nevralt-nett"
  | "transformer"
  | "multimodal"
  | "parametere"
  | "korpus"
  // frå tekst til tal
  | "token"
  | "bpe"
  | "vokabular"
  | "embedding"
  | "posisjonskoding"
  | "n-gram"
  | "oppslagstabell"
  | "hashfunksjon"
  | "hashkollisjon"
  | "ngram-minne"
  // inni transformeren
  | "attention"
  | "multi-head"
  | "kausal-maskering"
  | "softmax"
  | "layernorm"
  | "pre-norm"
  | "residual"
  | "feed-forward"
  | "gelu"
  | "situ-glu"
  | "logits"
  // læringa
  | "tap"
  | "perplexity"
  | "ablasjon"
  | "heldout"
  | "overtilpassing"
  | "backpropagation"
  | "autograd"
  | "gradient-clipping"
  | "minibatch"
  | "seq-len"
  // optimalisering
  | "laeringsrate"
  | "adam"
  | "muon"
  | "newton-schulz"
  | "nedtrapping"
  | "warmup"
  // tal og minne
  | "vekter"
  | "tensor"
  | "matmul"
  | "float32array"
  | "presisjon"
  | "kvantisering"
  // når modellen skriv
  | "temperatur"
  | "top-k"
  // finpussing
  | "sft"
  | "rlhf"
  | "dpo"
  // skala
  | "moe"
  | "flops"
  | "skalalov"
  | "gguf";

/* Gløymer du ei målform på eit ord, kompilerer det ikkje.
   Rekkjefølgja innanfor kvart tema er rekkjefølgja i ordlista. */
export const ORDLISTE: Record<OrdId, Oppslag> = {
  /* ---------------------------- grunnlaget ----------------------------- */
  spraakmodell: {
    tema: "grunnlag",
    bm: {
      vis: "språkmodell",
      ord: ["språkmodell", "språkmodellen", "språkmodeller", "språkmodellene"],
      def: "Et dataprogram trent til én ting: å gjette neste tegn eller ord i en tekst. All skrivingen er bare gjetting, om og om igjen.",
    },
    nn: {
      vis: "språkmodell",
      ord: ["språkmodell", "språkmodellen", "språkmodellar", "språkmodellane"],
      def: "Eit dataprogram trena til éin ting: å gjette neste teikn eller ord i ein tekst. All skrivinga er berre gjetting, om att og om att.",
    },
  },
  "nevralt-nett": {
    tema: "grunnlag",
    bm: {
      vis: "nevralt nett",
      ord: ["nevralt nett", "nevrale nett"],
      def: "Et nettverk av enkle regneenheter med justerbare vekter imellom – grunnmuren i moderne maskinlæring.",
    },
    nn: {
      vis: "nevralt nett",
      ord: ["nevralt nett", "nevrale nett"],
      def: "Eit nettverk av enkle rekneeiningar med justerbare vekter imellom – grunnmuren i moderne maskinlæring.",
    },
  },
  transformer: {
    tema: "grunnlag",
    bm: {
      vis: "transformer",
      ord: ["transformer", "transformeren"],
      def: "Byggetegningen bak moderne språkmodeller (fra 2017): lag på lag som først ser på sammenhengen i teksten, så tenker videre på den.",
    },
    nn: {
      vis: "transformer",
      ord: ["transformer", "transformeren"],
      def: "Byggjeteikninga bak moderne språkmodellar (frå 2017): lag på lag som først ser på samanhengen i teksten, så tenkjer vidare på han.",
    },
  },
  multimodal: {
    tema: "grunnlag",
    bm: {
      vis: "multimodal",
      ord: ["multimodal", "multimodale"],
      def: "En modell som kan arbeide med mer enn én type innhold, for eksempel tekst, bilder eller lyd. Hver type må først gjøres om til tallrepresentasjoner modellen forstår.",
    },
    nn: {
      vis: "multimodal",
      ord: ["multimodal", "multimodale"],
      def: "Ein modell som kan arbeida med meir enn éin type innhald, til dømes tekst, bilete eller lyd. Kvar type må først gjerast om til talrepresentasjonar modellen forstår.",
    },
  },
  parametere: {
    tema: "grunnlag",
    bm: {
      vis: "parametere",
      ord: ["parametere", "parametre"],
      def: "Samlenavn på alle tallene modellen lærer (vektene). Flere parametere = større modell – og mer regning.",
    },
    nn: {
      vis: "parametrar",
      ord: ["parametrar", "parametrane"],
      def: "Samlenamn på alle tala modellen lærer (vektene). Fleire parametrar = større modell – og meir rekning.",
    },
  },
  korpus: {
    tema: "grunnlag",
    bm: {
      vis: "korpus",
      ord: ["korpus", "korpuset"],
      def: "Tekstsamlingen modellen lærer av. Modellen blir som korpuset sitt – derfor betyr kvaliteten alt.",
    },
    nn: {
      vis: "korpus",
      ord: ["korpus", "korpuset"],
      def: "Tekstsamlinga modellen lærer av. Modellen blir som korpuset sitt – difor tyder kvaliteten alt.",
    },
  },

  /* -------------------------- frå tekst til tal ------------------------ */
  token: {
    tema: "tekst",
    bm: {
      vis: "token",
      ord: ["token", "tokenisering"],
      def: "Den minste tekstbiten modellen regner med – her ett enkelt tegn, i store modeller en ord-bit. Hvert token får sitt eget nummer.",
    },
    nn: {
      vis: "token",
      ord: ["token", "tokenisering"],
      def: "Den minste tekstbiten modellen reknar med – her eitt enkelt teikn, i store modellar ein ord-bit. Kvart token får sitt eige nummer.",
    },
  },
  bpe: {
    tema: "tekst",
    bm: {
      vis: "byte-pair encoding (BPE)",
      ord: ["byte-pair encoding", "BPE"],
      def: "Den vanligste måten å lage token på: start med enkelttegn og slå sammen det hyppigste paret, om og om igjen, til vanlige ord blir én bit. Prøv det selv i steg 3.",
    },
    nn: {
      vis: "byte-pair encoding (BPE)",
      ord: ["byte-pair encoding", "BPE"],
      def: "Den vanlegaste måten å lage token på: start med einskildteikn og slå saman det hyppigaste paret, om att og om att, til vanlege ord blir éin bit. Prøv det sjølv i steg 3.",
    },
  },
  vokabular: {
    tema: "tekst",
    bm: {
      vis: "vokabular (vocab)",
      ord: ["vokabular", "vokabularet"],
      def: "Lista over alle token modellen kjenner – her: hele tegnsettet. Alt modellen noensinne kan skrive, står her.",
    },
    nn: {
      vis: "vokabular (vocab)",
      ord: ["vokabular", "vokabularet"],
      def: "Lista over alle token modellen kjenner – her: heile teiknsettet. Alt modellen nokosinne kan skrive, står her.",
    },
  },
  embedding: {
    tema: "tekst",
    bm: {
      vis: "embedding",
      ord: ["embedding", "innbygging", "innbyggingen"],
      def: "Oversettelsen fra tegn til en liste med tall (en vektor) som modellen kan regne med. Tegn som ligner, får liknende tall.",
    },
    nn: {
      vis: "embedding",
      ord: ["embedding", "innbygging", "innbygginga"],
      def: "Omsetjinga frå teikn til ei liste med tal (ein vektor) som modellen kan rekne med. Teikn som liknar, får liknande tal.",
    },
  },
  posisjonskoding: {
    tema: "tekst",
    bm: {
      vis: "posisjonskoding",
      ord: ["posisjonskoding", "posisjonskodingen"],
      def: "Tegnene må vite hvor i setningen de står – «rev jager mus» og «mus jager rev» har de samme tegnene. Denne appen lærer én posisjonsvektor per plass og legger den til embeddingen.",
    },
    nn: {
      vis: "posisjonskoding",
      ord: ["posisjonskoding", "posisjonskodinga"],
      def: "Teikna må vite kvar i setninga dei står – «rev jagar mus» og «mus jagar rev» har dei same teikna. Denne appen lærer éin posisjonsvektor per plass og legg han til embeddinga.",
    },
  },
  "n-gram": {
    tema: "tekst",
    bm: {
      vis: "n-gram",
      ord: ["n-gram", "n-grammet", "n-grammene", "trigram"],
      def: "En sammenhengende rekke på n token. Et trigram har tre. Siden modellens token er enkelttegn, er «and» ett tegn-trigram her – ikke ett nytt token eller en ord-bit.",
    },
    nn: {
      vis: "n-gram",
      ord: ["n-gram", "n-grammet", "n-gramma", "trigram"],
      def: "Ei samanhengande rekkje på n token. Eit trigram har tre. Sidan tokena til modellen er einskildteikn, er «and» eitt teikn-trigram her – ikkje eitt nytt token eller ein ord-bit.",
    },
  },
  oppslagstabell: {
    tema: "tekst",
    bm: {
      vis: "oppslagstabell",
      ord: ["oppslagstabell", "oppslagstabellen"],
      def: "En samling lærbare rader der en nøkkel velger raden direkte. Modellen regner ikke gjennom hele tabellen; for hvert tegn henter trigramminnet bare én rad med dim tall.",
    },
    nn: {
      vis: "oppslagstabell",
      ord: ["oppslagstabell", "oppslagstabellen"],
      def: "Ei samling lærbare rader der ein nøkkel vel rada direkte. Modellen reknar ikkje gjennom heile tabellen; for kvart teikn hentar trigramminnet berre éi rad med dim tal.",
    },
  },
  hashfunksjon: {
    tema: "tekst",
    bm: {
      vis: "hashfunksjon",
      ord: ["hashfunksjon", "hashfunksjonen", "hashes", "hashing", "hashen"],
      def: "En deterministisk oppskrift som presser en stor nøkkelverden inn i et fast antall skuffer: samme trigram får alltid samme skuff. Denne appen bruker FNV-1a; Qwen-kildene spesifiserer ikke FNV.",
    },
    nn: {
      vis: "hashfunksjon",
      ord: ["hashfunksjon", "hashfunksjonen", "hasha", "hashing", "hashen"],
      def: "Ei deterministisk oppskrift som pressar ei stor nøkkelverd inn i eit fast tal skuffer: same trigram får alltid same skuff. Denne appen brukar FNV-1a; Qwen-kjeldene spesifiserer ikkje FNV.",
    },
  },
  hashkollisjon: {
    tema: "tekst",
    bm: {
      vis: "hashkollisjon",
      ord: ["hashkollisjon", "hashkollisjonen", "hashkollisjoner", "kollisjoner"],
      def: "Når to ulike trigram havner i samme skuff og derfor må dele minnerad. Det er ikke en programfeil, men prisen for en liten tabell: flere kollisjoner gir mindre minne og mer deling.",
    },
    nn: {
      vis: "hashkollisjon",
      ord: ["hashkollisjon", "hashkollisjonen", "hashkollisjonar", "kollisjonar"],
      def: "Når to ulike trigram hamnar i same skuff og difor må dela minnerad. Det er ikkje ein programfeil, men prisen for ein liten tabell: fleire kollisjonar gir mindre minne og meir deling.",
    },
  },
  "ngram-minne": {
    tema: "tekst",
    bm: {
      vis: "trigramminne",
      ord: ["trigramminne", "trigramminnet", "trigramoppslag", "trigramoppslagene"],
      def: "Qwen3.8-Flash-Next bruker korte token-n-gram som deterministiske nøkler til enorme minnetabeller ved lag 2 (Qwen Team, 2026, §2.3; full kilde i steg 10). Her er ideen krympet til én direkte FNV-tabell med 256 rader. Tre tegn er fortsatt tre token.",
    },
    nn: {
      vis: "trigramminne",
      ord: ["trigramminne", "trigramminnet", "trigramoppslag", "trigramoppslaga"],
      def: "Qwen3.8-Flash-Next brukar korte token-n-gram som deterministiske nøklar til enorme minnetabellar ved lag 2 (Qwen Team, 2026, §2.3; full kjelde i steg 10). Her er ideen krympa til éi direkte FNV-tabell med 256 rader. Tre teikn er framleis tre token.",
    },
  },

  /* ------------------------- inni transformeren ------------------------ */
  attention: {
    tema: "inni",
    bm: {
      vis: "selvoppmerksomhet (self-attention)",
      ord: ["multi-head oppmerksomhet", "selvoppmerksomhet", "oppmerksomhet", "oppmerksomheten"],
      def: "Mekanismen (attention) der hvert tegn ser tilbake på teksten før seg og plukker ut det som er viktig akkurat nå. Hjertet i transformeren.",
    },
    nn: {
      vis: "sjølvmerksemd (self-attention)",
      ord: ["multi-head merksemd", "sjølvmerksemd", "merksemd", "merksemda"],
      def: "Mekanismen (attention) der kvart teikn ser attende på teksten før seg og plukkar ut det som er viktig akkurat no. Hjartet i transformeren.",
    },
  },
  "multi-head": {
    tema: "inni",
    bm: {
      vis: "multi-head",
      ord: ["multi-head"],
      def: "Flere oppmerksomhets-blikk side om side i samme lag, hvert med sine egne vekter – ett hode kan følge med på tegnet før, et annet på starten av ordet. Se dem i steg 6.",
    },
    nn: {
      vis: "multi-head",
      ord: ["multi-head"],
      def: "Fleire merksemds-blikk side om side i same lag, kvart med sine eigne vekter – eitt hovud kan følgje med på teiknet før, eit anna på starten av ordet. Sjå dei i steg 6.",
    },
  },
  "kausal-maskering": {
    tema: "inni",
    bm: {
      vis: "kausal maskering (causal mask)",
      ord: ["kausal maskering"],
      def: "Sperren som skjuler fremtiden: når modellen gjetter et tegn, ser den bare tegnene som kom før – aldri fasiten.",
    },
    nn: {
      vis: "kausal maskering (causal mask)",
      ord: ["kausal maskering"],
      def: "Sperra som gøymer framtida: når modellen gjettar eit teikn, ser han berre teikna som kom før – aldri fasiten.",
    },
  },
  softmax: {
    tema: "inni",
    bm: {
      vis: "softmax",
      ord: ["softmax"],
      def: "Regnestykket som gjør poengsummene om til sannsynligheter som til sammen blir 100 %. Størst poeng → størst sjanse.",
    },
    nn: {
      vis: "softmax",
      ord: ["softmax"],
      def: "Reknestykket som gjer poengsummane om til sannsyn som til saman blir 100 %. Størst poeng → størst sjanse.",
    },
  },
  layernorm: {
    tema: "inni",
    bm: {
      vis: "LayerNorm",
      ord: ["LayerNorm"],
      def: "En liten opprydding som skalerer tallene til passe størrelse mellom leddene, så treningen holder seg stabil.",
    },
    nn: {
      vis: "LayerNorm",
      ord: ["LayerNorm"],
      def: "Ei lita opprydding som skalerer tala til passe storleik mellom ledda, så treninga held seg stabil.",
    },
  },
  "pre-norm": {
    tema: "inni",
    bm: {
      vis: "pre-norm",
      ord: ["pre-norm"],
      def: "Å legge LayerNorm før hvert ledd i stedet for etter. En liten ombytting som gjør treningen merkbart mer stabil – slik gjør denne appen og de fleste moderne modeller det.",
    },
    nn: {
      vis: "pre-norm",
      ord: ["pre-norm"],
      def: "Å leggje LayerNorm før kvart ledd i staden for etter. Ei lita ombyting som gjer treninga merkbart meir stabil – slik gjer denne appen og dei fleste moderne modellane det.",
    },
  },
  residual: {
    tema: "inni",
    bm: {
      vis: "residualveier",
      ord: ["residualveier", "residual"],
      def: "En snarvei der signalet hopper forbi et ledd og legges til igjen etterpå. Da kan ingen blokk ødelegge det som allerede er lært.",
    },
    nn: {
      vis: "residualvegar",
      ord: ["residualvegar", "residual"],
      def: "Ein snarveg der signalet hoppar forbi eit ledd og blir lagt til att etterpå. Då kan inga blokk øydeleggje det som alt er lært.",
    },
  },
  "feed-forward": {
    tema: "inni",
    bm: {
      vis: "feed-forward",
      ord: ["feed-forward"],
      def: "Et lite nevralt nett inni hver blokk som bearbeider hvert tegn for seg, etter at oppmerksomheten har hentet inn sammenhengen.",
    },
    nn: {
      vis: "feed-forward",
      ord: ["feed-forward"],
      def: "Eit lite nevralt nett inni kvar blokk som arbeider vidare med kvart teikn for seg, etter at merksemda har henta inn samanhengen.",
    },
  },
  gelu: {
    tema: "inni",
    bm: {
      vis: "GELU",
      ord: ["GELU"],
      def: "En mye brukt aktiveringsfunksjon: en myk knekk som avgjør hvor mye av hvert signal som slipper videre i nettet.",
    },
    nn: {
      vis: "GELU",
      ord: ["GELU"],
      def: "Ein mykje brukt aktiveringsfunksjon: ein mjuk knekk som avgjer kor mykje av kvart signal som slepp vidare i nettet.",
    },
  },
  "situ-glu": {
    tema: "inni",
    bm: {
      vis: "SiTU-GLU",
      ord: ["SiTU-GLU"],
      def: "En portfunksjon fra Kimi K3-oppskriften: laget styrer selv hvor mye som slipper gjennom, og et innebygd tak hindrer tallene i å løpe løpsk.",
    },
    nn: {
      vis: "SiTU-GLU",
      ord: ["SiTU-GLU"],
      def: "Ein portfunksjon frå Kimi K3-oppskrifta: laget styrer sjølv kor mykje som slepp gjennom, og eit innebygd tak hindrar tala i å renne løpsk.",
    },
  },
  logits: {
    tema: "inni",
    bm: {
      vis: "logits",
      ord: ["logits"],
      def: "Rå poengsum for hvert mulige neste tegn – før softmax gjør dem om til sannsynlighet.",
    },
    nn: {
      vis: "logits",
      ord: ["logits"],
      def: "Rå poengsum for kvart moglege neste teikn – før softmax gjer dei om til sannsyn.",
    },
  },

  /* ------------------------------ læringa ------------------------------ */
  tap: {
    tema: "laering",
    bm: {
      vis: "tap (cross-entropy loss)",
      ord: ["tap (loss)", "tap", "tapet", "kryssentropi", "cross-entropy"],
      def: "Feilmålet i treningen: hvor overrasket modellen ble av det riktige neste tegnet. Lavere tap = bedre gjetting. Regnestykket bak heter kryssentropi (cross-entropy).",
    },
    nn: {
      vis: "tap (cross-entropy loss)",
      ord: ["tap (loss)", "tap", "tapet", "kryssentropi", "cross-entropy"],
      def: "Feilmålet i treninga: kor overraska modellen vart av det rette neste teiknet. Lågare tap = betre gjetting. Reknestykket bak heiter kryssentropi (cross-entropy).",
    },
  },
  perplexity: {
    tema: "laering",
    bm: {
      vis: "perpleksitet (perplexity)",
      ord: ["perpleksitet", "perplexity"],
      def: "Tapet i en annen skala: hvor mange tegn modellen i praksis nøler mellom. Perfekt gjetting gir 1; ren sjanse gir hele vokabularet.",
    },
    nn: {
      vis: "perpleksitet (perplexity)",
      ord: ["perpleksitet", "perplexity"],
      def: "Tapet i ein annan skala: kor mange teikn modellen i praksis nøler mellom. Perfekt gjetting gir 1; rein sjanse gir heile vokabularet.",
    },
  },
  ablasjon: {
    tema: "laering",
    bm: {
      vis: "ablasjon",
      ord: ["ablasjon", "ablasjonen", "ablasjonsmåling"],
      def: "Et kontrollert forsøk der én del slås av eller på mens alt annet holdes likt. Trigramtesten bruker samme startvekter, tekstbiter og frø på begge sider, så forskjellen kan tilskrives minnet.",
    },
    nn: {
      vis: "ablasjon",
      ord: ["ablasjon", "ablasjonen", "ablasjonsmåling"],
      def: "Eit kontrollert forsøk der éin del blir slått av eller på medan alt anna er likt. Trigramtesten brukar same startvekter, tekstbitar og frø på båe sider, så skilnaden kan skrivast på minnet.",
    },
  },
  heldout: {
    tema: "laering",
    bm: {
      vis: "held-out-data",
      ord: ["held-out", "held-out-data", "testdata"],
      def: "Tekst modellen aldri får trene på, men som brukes til måling etterpå. Den viser om modellen har lært et mønster som virker videre, ikke bare pugget treningskorpuset.",
    },
    nn: {
      vis: "held-out-data",
      ord: ["held-out", "held-out-data", "testdata"],
      def: "Tekst modellen aldri får trena på, men som blir brukt til måling etterpå. Han viser om modellen har lært eit mønster som verkar vidare, ikkje berre pugga treningskorpuset.",
    },
  },
  overtilpassing: {
    tema: "laering",
    bm: {
      vis: "overtilpasning (overfitting)",
      ord: ["overtilpasning", "overtilpasset", "overfitting", "memoriserer"],
      def: "Når modellen blir svært god på teksten den trente på, men dårligere på ny tekst. Trigramminnet viser dette her: treningstapet faller kraftig samtidig som held-out-tapet stiger.",
    },
    nn: {
      vis: "overtilpassing (overfitting)",
      ord: ["overtilpassing", "overtilpassa", "overfitting", "memorerer"],
      def: "Når modellen blir svært god på teksten han trena på, men dårlegare på ny tekst. Trigramminnet viser dette her: treningstapet fell kraftig samstundes som held-out-tapet stig.",
    },
  },
  backpropagation: {
    tema: "laering",
    bm: {
      vis: "backpropagation",
      ord: ["backpropagation"],
      def: "Regnemetoden som sporer feilen bakover gjennom alle lagene, så hver vekt får vite hvilken vei den skal vris. På norsk: tilbakeføring av feil.",
    },
    nn: {
      vis: "backpropagation",
      ord: ["backpropagation"],
      def: "Reknemetoden som sporar feilen bakover gjennom alle laga, så kvar vekt får vite kva veg ho skal vridast. På norsk: tilbakeføring av feil.",
    },
  },
  autograd: {
    tema: "laering",
    bm: {
      vis: "autograd",
      ord: ["autograd"],
      def: "Maskineriet som automatisk regner ut hvilken vei hver vekt må vris – det som gjør backpropagation mulig å skrive.",
    },
    nn: {
      vis: "autograd",
      ord: ["autograd"],
      def: "Maskineriet som automatisk reknar ut kva veg kvar vekt må vridast – det som gjer backpropagation mogleg å skrive.",
    },
  },
  "gradient-clipping": {
    tema: "laering",
    bm: {
      vis: "gradient clipping",
      ord: ["gradient clipping", "gradientklipping"],
      def: "Et tak på hvor stor den samlede justeringen får bli i ett steg: blir feilsignalet for voldsomt, skaleres hele ned. Én vill bom får dermed ikke velte treningen.",
    },
    nn: {
      vis: "gradient clipping",
      ord: ["gradient clipping", "gradientklipping"],
      def: "Eit tak på kor stor den samla justeringa får bli i eitt steg: blir feilsignalet for valdsamt, blir heile skalert ned. Éin vill bom får dermed ikkje velte treninga.",
    },
  },
  minibatch: {
    tema: "laering",
    bm: {
      vis: "minibatch",
      ord: ["minibatch"],
      def: "En liten bunke tekstbiter modellen øver på i samme steg. Snittet av flere feil gir en jevnere vridning enn én bit alene.",
    },
    nn: {
      vis: "minibatch",
      ord: ["minibatch"],
      def: "Ein liten bunke tekstbitar modellen øver på i same steg. Snittet av fleire feil gir ei jamnare vriding enn éin bit åleine.",
    },
  },
  "seq-len": {
    tema: "laering",
    bm: {
      vis: "seq_len (kontekstvindu)",
      ord: ["seq_len", "kontekstvindu", "kontekstvinduet"],
      def: "Hvor mange tegn modellen ser på om gangen. Her 32–48 tegn; de store modellene husker hundretusenvis av token.",
    },
    nn: {
      vis: "seq_len (kontekstvindauge)",
      ord: ["seq_len", "kontekstvindauge", "kontekstvindauget"],
      def: "Kor mange teikn modellen ser på om gongen. Her 32–48 teikn; dei store modellane hugsar hundretusenvis av token.",
    },
  },

  /* --------------------------- optimalisering -------------------------- */
  laeringsrate: {
    tema: "optimalisering",
    bm: {
      vis: "læringsrate",
      ord: ["læringsrate", "læringsraten"],
      def: "Hvor mye skruene vris per steg. For mye: læringen kollapser i kaos. For lite: det tar evigheter.",
    },
    nn: {
      vis: "læringsrate",
      ord: ["læringsrate", "læringsraten"],
      def: "Kor mykje skruane blir vridne per steg. For mykje: læringa kollapsar i kaos. For lite: det tek all verdas tid.",
    },
  },
  adam: {
    tema: "optimalisering",
    bm: {
      vis: "Adam",
      ord: ["Adam"],
      def: "Arbeidshesten blant optimalisatorer siden 2015: gir hver enkelt vekt sin egen, selvjusterende skrittlengde.",
    },
    nn: {
      vis: "Adam",
      ord: ["Adam"],
      def: "Arbeidshesten blant optimalisatorar sidan 2015: gir kvar einskild vekt si eiga, sjølvjusterande skrittlengd.",
    },
  },
  muon: {
    tema: "optimalisering",
    bm: {
      vis: "Muon",
      ord: ["Muon"],
      def: "En ny optimalisator (2024) som vrir en hel tallmatrise under ett og jevner ut retningene. Brukes på de største modellene i dag.",
    },
    nn: {
      vis: "Muon",
      ord: ["Muon"],
      def: "Ein ny optimalisator (2024) som vrir ei heil talmatrise under eitt og jamnar ut retningane. Blir brukt på dei største modellane i dag.",
    },
  },
  "newton-schulz": {
    tema: "optimalisering",
    bm: {
      vis: "Newton–Schulz",
      ord: ["Newton–Schulz", "Newton-Schulz"],
      def: "Regnetrikset inni Muon: noen få runder matriseregning som jevner ut justeringen, så ingen enkelt retning får dominere steget.",
    },
    nn: {
      vis: "Newton–Schulz",
      ord: ["Newton–Schulz", "Newton-Schulz"],
      def: "Reknetrikset inni Muon: nokre få rundar matriserekning som jamnar ut justeringa, så inga enkelt retning får dominere steget.",
    },
  },
  nedtrapping: {
    tema: "optimalisering",
    bm: {
      vis: "kosinus-nedtrapping (cosine schedule)",
      ord: ["kosinus-nedtrapping", "cosine schedule", "nedtrapping", "nedtrappingen"],
      def: "Å senke læringsraten i en myk kosinusbue mot slutten av treningen. Standard for de store modellene – men prøv bryteren i steg 5: på denne lille teksten taper du som regel på det.",
    },
    nn: {
      vis: "kosinus-nedtrapping (cosine schedule)",
      ord: ["kosinus-nedtrapping", "cosine schedule", "nedtrapping", "nedtrappinga"],
      def: "Å senke læringsraten i ein mjuk kosinusboge mot slutten av treninga. Standard for dei store modellane – men prøv brytaren i steg 5: på denne vesle teksten taper du som regel på det.",
    },
  },
  warmup: {
    tema: "optimalisering",
    bm: {
      vis: "oppvarming (warmup)",
      ord: ["warmup", "oppvarming"],
      def: "Å starte med bitte liten læringsrate den første prosenten av treningen, mens alt inni modellen fortsatt er tilfeldig. Nedtrappingen her begynner slik.",
    },
    nn: {
      vis: "oppvarming (warmup)",
      ord: ["warmup", "oppvarming"],
      def: "Å starte med bitte lita læringsrate den første prosenten av treninga, medan alt inni modellen framleis er tilfeldig. Nedtrappinga her byrjar slik.",
    },
  },

  /* ---------------------------- tal og minne --------------------------- */
  vekter: {
    tema: "tal",
    bm: {
      vis: "vekter (weights)",
      ord: ["vektene", "vekter"],
      def: "Tallene inni modellen som lagrer alt den har lært. Å trene = å justere disse tallene bitte litt, tusenvis av ganger.",
    },
    nn: {
      vis: "vekter (weights)",
      ord: ["vektene", "vektane", "vekter"],
      def: "Tala inni modellen som lagrar alt han har lært. Å trene = å justere desse tala bitte litt, tusenvis av gonger.",
    },
  },
  tensor: {
    tema: "tal",
    bm: {
      vis: "tensor",
      ord: ["tensor", "tensorer", "tensorene"],
      def: "En tabell med tall – byggeklossen alt i modellen er laget av. Vektene, tegnene underveis og feilsignalene er alle tensorer.",
    },
    nn: {
      vis: "tensor",
      ord: ["tensor", "tensorar", "tensorane"],
      def: "Ein tabell med tal – byggjeklossen alt i modellen er laga av. Vektene, teikna undervegs og feilsignala er alle tensorar.",
    },
  },
  matmul: {
    tema: "tal",
    bm: {
      vis: "matmul (matrisemultiplikasjon)",
      ord: ["matmul", "matrisemultiplikasjon"],
      def: "Å gange to talltabeller sammen. Det er dette datamaskinen bruker nesten all tiden på – både her og i datasentrene.",
    },
    nn: {
      vis: "matmul (matrisemultiplikasjon)",
      ord: ["matmul", "matrisemultiplikasjon"],
      def: "Å gange to taltabellar saman. Det er dette datamaskina brukar nesten all tida på – både her og i datasentera.",
    },
  },
  float32array: {
    tema: "tal",
    bm: {
      vis: "Float32Array",
      ord: ["Float32Array"],
      def: "JavaScripts råe talltabell for 32-bits desimaltall. Hele regnemotoren i denne appen (src/lib/ml.ts) er bygd på den.",
    },
    nn: {
      vis: "Float32Array",
      ord: ["Float32Array"],
      def: "JavaScripts råe taltabell for 32-bits desimaltal. Heile reknemotoren i denne appen (src/lib/ml.ts) er bygd på han.",
    },
  },
  presisjon: {
    tema: "tal",
    bm: {
      vis: "fp32 / bf16 / MXFP4",
      ord: ["MXFP4", "bf16", "fp32"],
      def: "Hvor mange bits hvert tall får: 32, 16 eller 4. Færre bits betyr mindre minne og raskere regning, men grovere tall – i MXFP4 har hvert tall bare 16 mulige verdier.",
    },
    nn: {
      vis: "fp32 / bf16 / MXFP4",
      ord: ["MXFP4", "bf16", "fp32"],
      def: "Kor mange bits kvart tal får: 32, 16 eller 4. Færre bits tyder mindre minne og raskare rekning, men grovare tal – i MXFP4 har kvart tal berre 16 moglege verdiar.",
    },
  },
  kvantisering: {
    tema: "tal",
    bm: {
      vis: "kvantisering",
      ord: ["kvantisering", "kvantiseringen", "kvantisert"],
      def: "Å krympe en ferdig trent modell til færre bits per tall. Slankekuren i steg 5 gjør akkurat dette – det brede laget ned til 4 bits – så du ser hva det koster i tap.",
    },
    nn: {
      vis: "kvantisering",
      ord: ["kvantisering", "kvantiseringa", "kvantisert"],
      def: "Å krympe ein ferdig trena modell til færre bits per tal. Slankekuren i steg 5 gjer akkurat dette – det breie laget ned til 4 bits – så du ser kva det kostar i tap.",
    },
  },

  /* -------------------------- når modellen skriv ----------------------- */
  temperatur: {
    tema: "skriv",
    bm: {
      vis: "temperatur",
      ord: ["temperatur", "temperaturen"],
      def: "Styrer sjansespillet når neste tegn velges: 0 = alltid det sikreste tegnet, høyere = jevnere lodd og villere tekst.",
    },
    nn: {
      vis: "temperatur",
      ord: ["temperatur", "temperaturen"],
      def: "Styrer sjansespelet når neste teikn blir valt: 0 = alltid det sikraste teiknet, høgare = jamnare lodd og villare tekst.",
    },
  },
  "top-k": {
    tema: "skriv",
    bm: {
      vis: "top-k",
      ord: ["top-k"],
      def: "Modellen får bare trekke blant de k mest sannsynlige tegnene – resten kastes før loddtrekningen.",
    },
    nn: {
      vis: "top-k",
      ord: ["top-k"],
      def: "Modellen får berre trekkje mellom dei k mest sannsynlege teikna – resten blir kasta før loddtrekkinga.",
    },
  },

  /* ----------------------------- finpussing ---------------------------- */
  sft: {
    tema: "finpuss",
    bm: {
      vis: "instruksjonstrening (SFT)",
      ord: ["instruksjonstrening (SFT)", "instruksjonstrening", "SFT"],
      def: "«Supervised Fine-Tuning»: Modellen trenes videre på instruksjoner med gode fasitsvar, slik at en tekstfortsetter lærer mønsteret spørsmål inn, hjelpsomt svar ut.",
    },
    nn: {
      vis: "instruksjonstrening (SFT)",
      ord: ["instruksjonstrening (SFT)", "instruksjonstrening", "SFT"],
      def: "«Supervised Fine-Tuning»: Modellen blir trena vidare på instruksjonar med gode fasitsvar, slik at ein tekstframhaldar lærer mønsteret spørsmål inn, hjelpsamt svar ut.",
    },
  },
  rlhf: {
    tema: "finpuss",
    bm: {
      vis: "RLHF",
      ord: ["RLHF"],
      def: "«Reinforcement Learning from Human Feedback»: mennesker velger de beste svarene, og modellen justeres mot det folk foretrekker. Slik lærte chatbotene folkeskikk.",
    },
    nn: {
      vis: "RLHF",
      ord: ["RLHF"],
      def: "«Reinforcement Learning from Human Feedback»: menneske vel dei beste svara, og modellen blir justert mot det folk føretrekkjer. Slik lærte chatbotane folkeskikk.",
    },
  },
  dpo: {
    tema: "finpuss",
    bm: {
      vis: "DPO",
      ord: ["DPO"],
      def: "«Direct Preference Optimization»: en enkel RLHF-oppskrift som flytter modellen rett mot svarene du valgte – uten et eget nettverk som setter poeng på svarene.",
    },
    nn: {
      vis: "DPO",
      ord: ["DPO"],
      def: "«Direct Preference Optimization»: ei enkel RLHF-oppskrift som flyttar modellen rett mot svara du valde – utan eit eige nettverk som set poeng på svara.",
    },
  },

  /* -------------------------------- skala ------------------------------ */
  moe: {
    tema: "skala",
    bm: {
      vis: "mixture of experts (MoE)",
      ord: ["MoE"],
      def: "«Mixture of Experts»: det brede laget deles i flere smale eksperter, og bare noen få vekkes per tegn. Like mange vekter, mye mindre regning.",
    },
    nn: {
      vis: "mixture of experts (MoE)",
      ord: ["MoE"],
      def: "«Mixture of Experts»: det breie laget blir delt i fleire smale ekspertar, og berre nokre få blir vekte per teikn. Like mange vekter, mykje mindre rekning.",
    },
  },
  flops: {
    tema: "skala",
    bm: {
      vis: "FLOPs",
      ord: ["FLOPs"],
      def: "Antall regneoperasjoner på desimaltall – måleenheten for hva trening koster. De største treningskjøringene måles i billioner av billioner FLOPs.",
    },
    nn: {
      vis: "FLOPs",
      ord: ["FLOPs"],
      def: "Talet på rekneoperasjonar på desimaltal – måleeininga for kva trening kostar. Dei største treningskøyringane blir målte i billionar av billionar FLOPs.",
    },
  },
  skalalov: {
    tema: "skala",
    bm: {
      vis: "skalalover",
      ord: ["skalalover", "skalalov", "skaleringslover", "skaleringslov"],
      def: "Den påfallende jevne sammenhengen mellom regnekraft og resultat: ti ganger mer trening gir et fast, forutsigbart hakk lavere tap. Det er denne kurven som får folk til å bygge datasentre.",
    },
    nn: {
      vis: "skalalover",
      ord: ["skalalover", "skalalov", "skaleringslover", "skaleringslov"],
      def: "Den påfallande jamne samanhengen mellom reknekraft og resultat: ti gonger meir trening gir eit fast, føreseieleg hakk lågare tap. Det er denne kurva som får folk til å byggje datasenter.",
    },
  },
  gguf: {
    tema: "skala",
    bm: {
      vis: "GGUF",
      ord: ["GGUF"],
      def: "Filformatet åpne modeller deles i (kjent fra llama.cpp): én fil med alle vektene pluss en liten innholdsfortegnelse.",
    },
    nn: {
      vis: "GGUF",
      ord: ["GGUF"],
      def: "Filformatet opne modellar blir delte i (kjent frå llama.cpp): éi fil med alle vektene pluss ei lita innhaldsliste.",
    },
  },
};

/* --- Ordlista sortert etter tema (steg 11) ------------------------------ */

export interface OrdlisteGruppe {
  tema: Tema;
  oppslag: { id: OrdId; vis: string; def: string }[];
}

/** Alle oppslaga gruppert i temarekkjefølgje; rekkjefølgja innanfor kvart
 *  tema er rekkjefølgja i ORDLISTE. */
export function ordlisteTema(lang: Lang): OrdlisteGruppe[] {
  const ids = Object.keys(ORDLISTE) as OrdId[];
  return TEMA_REKKEFOLGJE.map((tema) => ({
    tema,
    oppslag: ids
      .filter((id) => ORDLISTE[id].tema === tema)
      .map((id) => ({ id, vis: ORDLISTE[id][lang].vis, def: ORDLISTE[id][lang].def })),
  }));
}

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
