// Norsk treningsdata på bokmål og nynorsk.
// Ei blanding av sjølvstendig skriven tekst på ulike tema: natur og geografi,
// folkeeventyr, kvardag, vasskrinslaupet, fiber og lys, ordtak, årstider og
// enkle spørsmål/svar. Teksten er eigenprodusert råtekst.
// NB: SEEDS i i18n.ts (t.d. «Det var en gang», «Norge er», «Jeg heter»,
// «Vann er» og nynorsk-variantane) må framleis finnast i tekstane.

export type CorpusLang = "bm" | "nn";

const bokmaal = `Norge er et land med lange kyster og høye fjell. Lengst i nord går sola aldri ned om sommeren, og lengst i sør blomstrer frukttrærne tidlig om våren. Mellom fjellene ligger daler med gårder og små tettsteder. Langs kysten finner du tusenvis av øyer og skjær. Folk har alltid levd av havet, skogen og jorda her. Toget går gjennom tunneler og over broer, og ferjene binder øyene sammen med fastlandet.

Det var en gang en fisker som rodde ut på fjorden hver eneste morgen. En kveld i storm hørte han noen rope svakt fra et skjær. Der satt en liten selunge, våt og redd. Fiskeren løftet den varsomt opp i båten og rodde den inn til viken der moren ventet. Året etter var fjorden full av fisk som aldri før, og folk sa at havet husker den som viser hjertelag.

Om morgenen våkner jeg av at lyset faller inn gjennom vinduet. Jeg spiser brødskiver med brunost og drikker melk før jeg sykler til skolen. Etter skolen spiller vi fotball på løkka til det blir mørkt. Om kvelden leser mamma høyt fra en bok mens regnet trommer på taket. Da er det godt å være hjemme.

Vann er alltid på reise. Regnet faller på fjellet, samler seg i bekker og renner ut i elver og hav. Sola varmer havet, vannet stiger opp som usynlig damp og blir til skyer. Skyene driver innover land og slipper regnet ned igjen. Slik går vannet i en evig ring, og det samme vannet har vært her siden jorda var ung.

Under bakken og langs havbunnen ligger lange kabler av glass. Inne i dem reiser lyset med meldinger mellom folk. Når du sender en hilsen til noen langt borte, blir ordene dine til små blink som farer gjennom kabelen fortere enn du rekker å blunke. Slik er hele landet knyttet sammen, fra de ytterste øyene til de innerste dalene.

Gamle ordtak bærer på mye visdom. Øvelse gjør mester. Borte bra, men hjemme best. Den som ler sist, ler best. Man skal ikke selge skinnet før bjørnen er skutt. Mange bekker små gjør en stor å. Det er ikke gull alt som glimrer. Bedre føre var enn etter snar.

Årstidene setter sitt preg på landet. Om vinteren ligger snøen tung på granene, og barna aker på akebrett ned bakkene. Om våren bruser elvene av smeltevann. Om sommeren bader vi i vannet til langt på kveld, og om høsten plukker vi bær og sopp i skogen. Været kan snu fort, derfor sier folk at det ikke finnes dårlig vær, bare dårlige klær.

Hva heter du? Jeg heter Kari og jeg bor ved havet. Hvor kommer du fra? Jeg kommer fra en liten bygd i fjellet. Hva liker du å gjøre? Jeg liker å lese bøker og å fiske med bestefar. Er det kaldt ute i dag? Ja, ta på deg lue og votter. Kan du vise meg veien til stasjonen? Ja, følg veien rett fram og ta til venstre ved kirken.`;

const nynorsk = `Noreg er eit land med lange kyster og høge fjell. Lengst i nord går ikkje sola ned om sommaren, og lengst i sør blømer frukttrea tidleg om våren. Mellom fjella ligg dalar med gardar og små tettstader. Langs kysten finn du tusenvis av øyar og skjer. Folk har alltid levd av havet, skogen og jorda her. Toget går gjennom tunnelar og over bruer, og ferjene bind øyane saman med fastlandet.

Det var ein gong ein fiskar som rodde ut på fjorden kvar einaste morgon. Ein kveld i storm høyrde han nokon rope svakt frå eit skjer. Der sat ein liten selunge, våt og redd. Fiskaren løfta han varsamt opp i båten og rodde han inn til vika der mora venta. Året etter var fjorden full av fisk som aldri før, og folk sa at havet hugsar den som viser hjartelag.

Om morgonen vaknar eg av at lyset fell inn gjennom vindauget. Eg et brødskiver med brunost og drikk mjølk før eg syklar til skulen. Etter skulen spelar vi fotball på løkka til det blir mørkt. Om kvelden les mamma høgt frå ei bok medan regnet trommar på taket. Då er det godt å vere heime.

Vatn er alltid på reise. Regnet fell på fjellet, samlar seg i bekkar og renn ut i elvar og hav. Sola varmar havet, vatnet stig opp som usynleg damp og blir til skyer. Skyene driv innover land og slepp regnet ned att. Slik går vatnet i ein evig ring, og det same vatnet har vore her sidan jorda var ung.

Under bakken og langs havbotnen ligg lange kablar av glas. Inne i dei reiser lyset med meldingar mellom folk. Når du sender ei helsing til nokon langt borte, blir orda dine til små blink som fer gjennom kabelen fortare enn du rekk å blunke. Slik er heile landet knytt saman, frå dei ytste øyane til dei inste dalane.

Gamle ordtak ber på mykje visdom. Øving gjer meister. Borte bra, men heime best. Den som ler sist, ler best. Ein skal ikkje selje skinnet før bjørnen er skoten. Mange bekkar små gjer ei stor å. Det er ikkje gull alt som glimrar. Betre føre var enn etter snar.

Årstidene set sitt preg på landet. Om vinteren ligg snøen tung på granene, og borna akar på akebrett ned bakkane. Om våren brusar elvane av smeltevatn. Om sommaren badar vi i vatnet til langt på kveld, og om hausten plukkar vi bær og sopp i skogen. Vêret kan snu fort, difor seier folk at det ikkje finst dårleg vêr, berre dårlege klede.

Kva heiter du? Eg heiter Kari og eg bur ved havet. Kvar kjem du frå? Eg kjem frå ei lita bygd i fjellet. Kva likar du å gjere? Eg likar å lese bøker og å fiske med bestefar. Er det kaldt ute i dag? Ja, ta på deg lue og vottar. Kan du vise meg vegen til stasjonen? Ja, følg vegen beint fram og ta til venstre ved kyrkja.`;

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
