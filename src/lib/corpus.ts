// Norsk nynorsk treningsdata.
// Dette er ei blanding av sjølvstendig skrive nynorsk tekst på ulike tema:
// natur, folkeeventyr, kvardag, ordtak og enkle spørsmål/svar.
// Teksten er offentleg / eigenprodusert og blir brukt som råtekst til modellen.

export const corpus = `Noreg er eit langt og smalt land i nord. Her finst det høge fjell, djupe fjordar og vide skogar. Om vinteren fell det mykje snø, og elvane kan fryse til is. Mange bygder ligg langs kysten, der folk lever av fiske og jordbruk. Om sommaren skinn sola lenge om kvelden, og nettene er korte. Bøndene dyrkar korn, poteter og grønsaker i grøne dalar. Fiskarane drar ut på havet tidleg kvar morgon for å fanga torsk og sild.

Det var ein gong ein gamal mann som budde aleine i ei lita hytte ved skogen. Kvar morgon gjekk han ut for å henta ved og vatn. Ein dag fann han ei lita fugl som hadde brote vengen. Han tok ho med seg inn og stelte vel med ho i mange dagar. Etter ei veke kunne fuglen flyga att, og ho kom attende kvart vår for å syngja for den gamle mannen. Han smilte og visste at venner kjem attende når ein er god mot dei.

Eg står tidleg opp kvar dag og et frukost før eg går på skulen. I klasserommet sit vi mange elevar i lag, og læraren forklarer matematikk og naturfag på tavla. I friminutta leikar vi ute i skulegarden til ringa går. Når eg kjem heim, gjer eg lekser og hjelper til heime. Om helga vitjar vi bestefar som bur i ein annan bygd. Ho lagar god mat og fortel gamle historier frå ungdomen.

Vatn er viktig for alt liv på jorda. Utan vatn kan korkje plantar, dyr eller menneske leva. Vi brukar vatn til å drikka, til å vaska oss og til å dyrka mat. Mange stader i verda er det for lite vatn, og folk må gå langt for å henta det. Difor må vi passa på å ikkje sløsa. Når vi børstar tennene, bør krana vere av.

Norske ordtak er fulle av gamal visdom. Betre sein enn aldri. Ein god dag kjem sjeldan aleine. Det er ikkje gull alt som glimrar. Mange små bekkar gjer ei stor elv. Ein skal ikkje skryta av dagen før kvelden kjem. Ute av auge, ute av sinne. Orda til gamle folk er ofte sanne.

Vêret i Noreg skiftar ofte. Ein dag skin sola, og dagen etter kan det regna eller snøa. Våren er mild og grøn, sommaren er lys og varm, hausten er fargerik, og vinteren er kvit og kald. Folk kler seg etter vêret og etter årstida. Mange likar best dagane når det er vindstille og klart.

Kva heiter du? Eg heiter Ola og eg kjem frå Noreg. Kva tid kjem toget? Det kjem klokka fire. Kor mykje kostar brødet? Brødet kostar tjue kroner. Kan du hjelpa meg? Ja, eg hjelper deg gjerne. Er det langt å gå? Nei, det er berre eit lite stykke. Likar du å lesa bøker? Ja, eg les kvelden lang.`;

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
