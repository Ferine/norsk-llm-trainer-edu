// Norsk treningsdata på bokmål og nynorsk.
// Ei blanding av sjølvstendig skriven tekst på ulike tema: natur, folkeeventyr,
// kvardag, ordtak og enkle spørsmål/svar. Teksten er eigenprodusert råtekst.

export type CorpusLang = "bm" | "nn";

const bokmaal = `Norge er et langt og smalt land i nord. Her finnes høye fjell, dype fjorder og store skoger. Om vinteren faller det mye snø, og elvene kan fryse til is. Mange bygder ligger langs kysten, der folk lever av fiske og jordbruk. Om sommeren skinner sola lenge om kvelden, og nettene er korte. Bøndene dyrker korn, poteter og grønnsaker i grønne daler. Fiskerne drar ut på havet tidlig hver morgen for å fange torsk og sild.

Det var en gang en gammel mann som bodde alene i en liten hytte ved skogen. Hver morgen gikk han ut for å hente ved og vann. En dag fant han en liten fugl som hadde brukket vingen. Han tok den med seg inn og stelte godt med den i mange dager. Etter en uke kunne fuglen fly igjen, og den kom tilbake hver vår for å synge for den gamle mannen. Han smilte og visste at venner kommer tilbake når man er god mot dem.

Jeg står tidlig opp hver dag og spiser frokost før jeg går på skolen. I klasserommet sitter vi mange elever sammen, og læreren forklarer matematikk og naturfag på tavla. I friminuttene leker vi ute i skolegården til det ringer inn. Når jeg kommer hjem, gjør jeg lekser og hjelper til hjemme. I helgene besøker vi bestefar som bor i en annen bygd. Han lager god mat og forteller gamle historier fra ungdommen.

Vann er viktig for alt liv på jorda. Uten vann kan verken planter, dyr eller mennesker leve. Vi bruker vann til å drikke, til å vaske oss og til å dyrke mat. Mange steder i verden er det for lite vann, og folk må gå langt for å hente det. Derfor må vi passe på å ikke sløse. Når vi pusser tennene, bør kranen være av.

Norske ordtak er fulle av gammel visdom. Bedre sent enn aldri. En god dag kommer sjelden alene. Det er ikke gull alt som glimrer. Mange bekker små gjør en stor å. Man skal ikke skryte av dagen før kvelden kommer. Ute av øye, ute av sinn. Ordene til gamle folk er ofte sanne.

Været i Norge skifter ofte. Den ene dagen skinner sola, og dagen etter kan det regne eller snø. Våren er mild og grønn, sommeren er lys og varm, høsten er fargerik, og vinteren er hvit og kald. Folk kler seg etter været og etter årstiden. Mange liker best dagene når det er vindstille og klart.

Hva heter du? Jeg heter Ola og jeg kommer fra Norge. Når kommer toget? Det kommer klokka fire. Hvor mye koster brødet? Brødet koster tjue kroner. Kan du hjelpe meg? Ja, jeg hjelper deg gjerne. Er det langt å gå? Nei, det er bare et lite stykke. Liker du å lese bøker? Ja, jeg leser hele kvelden.`;

const nynorsk = `Noreg er eit langt og smalt land i nord. Her finst det høge fjell, djupe fjordar og vide skogar. Om vinteren fell det mykje snø, og elvane kan fryse til is. Mange bygder ligg langs kysten, der folk lever av fiske og jordbruk. Om sommaran skinn sola lenge om kvelden, og nettene er korte. Bøndene dyrkar korn, poteter og grønsaker i grøne dalar. Fiskarane drar ut på havet tidleg kvar morgon for å fanga torsk og sild.

Det var ein gong ein gamal mann som budde aleine i ei lita hytte ved skogen. Kvar morgon gjekk han ut for å henta ved og vatn. Ein dag fann han ei lita fugl som hadde brote vengen. Han tok ho med seg inn og stelte vel med ho i mange dagar. Etter ei veke kunne fuglen flyga att, og ho kom attende kvart vår for å syngja for den gamle mannen. Han smilte og visste at venner kjem attende når ein er god mot dei.

Eg står tidleg opp kvar dag og et frukost før eg går på skulen. I klasserommet sit vi mange elevar i lag, og læraren forklarer matematikk og naturfag på tavla. I friminutta leikar vi ute i skulegarden til ringa går. Når eg kjem heim, gjer eg lekser og hjelper til heime. Om helga vitjar vi bestefar som bur i ein annan bygd. Ho lagar god mat og fortel gamle historier frå ungdomen.

Vatn er viktig for alt liv på jorda. Utan vatn kan korkje plantar, dyr eller menneske leva. Vi brukar vatn til å drikka, til å vaska oss og til å dyrka mat. Mange stader i verda er det for lite vatn, og folk må gå langt for å henta det. Difor må vi passa på å ikkje sløsa. Når vi børstar tennene, bør krana vere av.

Norske ordtak er fulle av gamal visdom. Betre sein enn aldri. Ein god dag kjem sjeldan aleine. Det er ikkje gull alt som glimrar. Mange små bekkar gjer ei stor elv. Ein skal ikkje skryta av dagen før kvelden kjem. Ute av auge, ute av sinne. Orda til gamle folk er ofte sanne.

Vêret i Noreg skiftar ofte. Ein dag skin sola, og dagen etter kan det regna eller snøa. Våren er mild og grøn, sommaran er lys og varm, hausten er fargerik, og vinteren er kvit og kald. Folk kler seg etter vêret og etter årstida. Mange likar best dagane når det er vindstille og klart.

Kva heiter du? Eg heiter Ola og eg kjem frå Noreg. Kva tid kjem toget? Det kjem klokka fire. Kor mykje kostar brødet? Brødet kostar tjue kroner. Kan du hjelpa meg? Ja, eg hjelper deg gjerne. Er det langt å gå? Nei, det er berre eit lite stykke. Likar du å lesa bøker? Ja, eg les kvelden lang.`;

export const corpora: Record<CorpusLang, string> = { bm: bokmaal, nn: nynorsk };

// Bakoverkompatibel standard-eksport (bokmål). Brukt av testane.
export const corpus = corpora.bm;

export interface Tokenizer {
  stoi: Record<string, number>;
  itos: string[];
  vocab: number;
  encode: (s: string) => number[];
  decode: (ids: number[]) => string;
}

// Teikn-nivå (character-level) tokenisering: kvart teikn er éin token.
export function buildTokenizer(text: string): Tokenizer {
  const chars = Array.from(new Set(text)).sort();
  const stoi: Record<string, number> = {};
  const itos: string[] = [];
  chars.forEach((c, i) => {
    stoi[c] = i;
    itos.push(c);
  });
  return {
    stoi,
    itos,
    vocab: chars.length,
    encode: (s: string) => {
      const out: number[] = [];
      for (const ch of s) {
        const id = stoi[ch];
        if (id !== undefined) out.push(id);
      }
      return out;
    },
    decode: (ids: number[]) => ids.map((i) => itos[i]).join(""),
  };
}
