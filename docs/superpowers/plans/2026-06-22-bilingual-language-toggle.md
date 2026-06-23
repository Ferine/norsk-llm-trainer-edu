# Bilingual UI (Bokmål default, Nynorsk optional) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Bokmål the default language of the in-browser transformer trainer while keeping Nynorsk as a one-click option, flipping the entire UI, the training corpus, and the generation seeds.

**Architecture:** A hand-rolled i18n module (`src/lib/i18n.ts`) holds a `Lang` type, both string bundles, and per-language seeds. `corpus.ts` gains a `corpora` map. `App.tsx` owns `lang` state (default `"bm"`, persisted to `localStorage`), renders a header toggle, resolves the active string bundle, and passes it to leaf components as an `s` prop. Switching language reuses the existing model-rebuild path.

**Tech Stack:** React 19, TypeScript 5.9, Vite 7, Tailwind 4, plain `.mjs` Node test scripts (no test framework). Zero runtime dependencies; bundled to a single file via `vite-plugin-singlefile`.

## Global Constraints

- No new runtime or dev dependencies (no i18n library). Hand-rolled bundles only. — verbatim from spec non-goals.
- Only two languages: `"bm"` (Bokmål) and `"nn"` (Nynorsk). Default `"bm"`.
- The Nynorsk bundle reuses the existing copy verbatim (move, don't reword). The Bokmål bundle is the fresh copy authored in this plan.
- No change to `src/lib/ml.ts`, the training/DPO algorithms, or Tailwind styling.
- `src/components/ui.tsx` is structural — do not modify it.
- Keep `export const corpus` in `corpus.ts` so the existing compiled tests keep working.
- Package manager is **pnpm** (`pnpm@10.28.2`). Use `pnpm`, never `npm`/`yarn`.
- Commit messages end with the project's trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Fresh Bokmål corpus + `corpora` map

**Files:**
- Modify: `src/lib/corpus.ts` (lines 1–18: header comment + `corpus` export)
- Create: `test/corpus-lang.test.mjs`
- Modify: `package.json` (the `test` script)

**Interfaces:**
- Produces: `export const corpora: Record<"bm" | "nn", string>`; `export const corpus: string` (= `corpora.bm`, unchanged name/type so existing tests compile).

- [ ] **Step 1: Write the failing test**

Create `test/corpus-lang.test.mjs`:

```js
import assert from "node:assert/strict";
import { corpus, corpora } from "./dist/corpus.js";

assert.ok(corpora.bm.length > 200, "bokmål corpus should be non-trivial");
assert.ok(corpora.nn.length > 200, "nynorsk corpus should be non-trivial");
assert.notEqual(corpora.bm, corpora.nn, "the two corpora must differ");
assert.equal(corpus, corpora.bm, "default corpus export must be bokmål");
// Bokmål markers absent from Nynorsk: "ikke" / "hvor" should be in bm, "ikkje"/"kor" in nn
assert.ok(corpora.bm.includes("ikke"), "bokmål uses 'ikke'");
assert.ok(corpora.nn.includes("ikkje"), "nynorsk uses 'ikkje'");

console.log("corpus-lang: OK");
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm run test:build && node test/corpus-lang.test.mjs`
Expected: FAIL — `corpora` is `undefined` (`Cannot read properties of undefined (reading 'bm')`).

- [ ] **Step 3: Implement — replace lines 1–18 of `src/lib/corpus.ts`**

Replace the header comment and the single `corpus` export with both corpora. The Nynorsk string is the **existing** corpus text moved verbatim; the Bokmål string is new:

```ts
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

const nynorsk = `Noreg er eit langt og smalt land i nord. Her finst det høge fjell, djupe fjordar og vide skogar. Om vinteren fell det mykje snø, og elvane kan fryse til is. Mange bygder ligg langs kysten, der folk lever av fiske og jordbruk. Om sommaren skinn sola lenge om kvelden, og nettene er korte. Bøndene dyrkar korn, poteter og grønsaker i grøne dalar. Fiskarane drar ut på havet tidleg kvar morgon for å fanga torsk og sild.

Det var ein gong ein gamal mann som budde aleine i ei lita hytte ved skogen. Kvar morgon gjekk han ut for å henta ved og vatn. Ein dag fann han ei lita fugl som hadde brote vengen. Han tok ho med seg inn og stelte vel med ho i mange dagar. Etter ei veke kunne fuglen flyga att, og ho kom attende kvart vår for å syngja for den gamle mannen. Han smilte og visste at venner kjem attende når ein er god mot dei.

Eg står tidleg opp kvar dag og et frukost før eg går på skulen. I klasserommet sit vi mange elevar i lag, og læraren forklarer matematikk og naturfag på tavla. I friminutta leikar vi ute i skulegarden til ringa går. Når eg kjem heim, gjer eg lekser og hjelper til heime. Om helga vitjar vi bestefar som bur i ein annan bygd. Ho lagar god mat og fortel gamle historier frå ungdomen.

Vatn er viktig for alt liv på jorda. Utan vatn kan korkje plantar, dyr eller menneske leva. Vi brukar vatn til å drikka, til å vaska oss og til å dyrka mat. Mange stader i verda er det for lite vatn, og folk må gå langt for å henta det. Difor må vi passa på å ikkje sløsa. Når vi børstar tennene, bør krana vere av.

Norske ordtak er fulle av gamal visdom. Betre sein enn aldri. Ein god dag kjem sjeldan aleine. Det er ikkje gull alt som glimrar. Mange små bekkar gjer ei stor elv. Ein skal ikkje skryta av dagen før kvelden kjem. Ute av auge, ute av sinne. Orda til gamle folk er ofte sanne.

Vêret i Noreg skiftar ofte. Ein dag skin sola, og dagen etter kan det regna eller snøa. Våren er mild og grøn, sommaren er lys og varm, hausten er fargerik, og vinteren er kvit og kald. Folk kler seg etter vêret og etter årstida. Mange likar best dagane når det er vindstille og klart.

Kva heiter du? Eg heiter Ola og eg kjem frå Noreg. Kva tid kjem toget? Det kjem klokka fire. Kor mykje kostar brødet? Brødet kostar tjue kroner. Kan du hjelpa meg? Ja, eg hjelper deg gjerne. Er det langt å gå? Nei, det er berre eit lite stykke. Likar du å lesa bøker? Ja, eg les kvelden lang.`;

export const corpora: Record<CorpusLang, string> = { bm: bokmaal, nn: nynorsk };

// Bakoverkompatibel standard-eksport (bokmål). Brukt av testane.
export const corpus = corpora.bm;
```

Leave everything from `export interface Tokenizer` downward unchanged.

- [ ] **Step 4: Wire the new test into `package.json`**

In the `test` script, append the new test after `clone.test.mjs` (order is not important; keep the chain). Change:

```
"test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs"
```
to:
```
"test": "pnpm run test:build && node test/seq-logprob.test.mjs && node test/dpo-loss.test.mjs && node test/clone.test.mjs && node test/corpus-lang.test.mjs && node test/generate-parity.test.mjs && node test/dpo-smoke.test.mjs"
```

- [ ] **Step 5: Run the full suite to verify green**

Run: `pnpm test`
Expected: all tests pass, including `corpus-lang: OK`. (`generate-parity` and `dpo-smoke` now run against the Bokmål default — they assert structural properties, not specific words, so they stay green.)

- [ ] **Step 6: Commit**

```bash
git add src/lib/corpus.ts test/corpus-lang.test.mjs package.json
git commit -m "feat(i18n): add Bokmål corpus and corpora map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: i18n module (`src/lib/i18n.ts`) + parity test

**Files:**
- Create: `src/lib/i18n.ts`
- Create: `test/i18n-parity.test.mjs`
- Modify: `package.json` (`test:build` to also compile `i18n.ts`; `test` to run the parity test)

**Interfaces:**
- Produces:
  - `export type Lang = "bm" | "nn"`
  - `export const LANGS: { id: Lang; label: string; htmlLang: string; locale: string }[]`
  - `export interface Strings` (nested groups: `header, hero, understand, data, arch, train, chat, rlhf, warning, extra, lossLast, docTitle`)
  - `export const STRINGS: Record<Lang, Strings>`
  - `export interface Seeds { chatPrompt: string; examples: string[]; sampleSentence: string; trainSeed: string }`
  - `export const SEEDS: Record<Lang, Seeds>`
- Interpolated strings are **functions** (e.g. `minibatch: (n: number) => string`). Parity test treats functions as leaves.

- [ ] **Step 1: Write the failing parity test**

Create `test/i18n-parity.test.mjs`:

```js
import assert from "node:assert/strict";
import { STRINGS, SEEDS, LANGS } from "./dist/i18n.js";

// recursive key-shape comparison; functions and strings are leaves
function shape(v) {
  if (Array.isArray(v)) return v.map(shape);
  if (v && typeof v === "object") {
    const o = {};
    for (const k of Object.keys(v).sort()) o[k] = shape(v[k]);
    return o;
  }
  return typeof v; // "string" | "function" | "number"
}

assert.deepEqual(shape(STRINGS.bm), shape(STRINGS.nn), "STRINGS bm/nn key shapes must match");
assert.deepEqual(shape(SEEDS.bm), shape(SEEDS.nn), "SEEDS bm/nn key shapes must match");
assert.equal(LANGS.length, 2);
assert.equal(LANGS[0].id, "bm", "Bokmål must be first (default)");
assert.deepEqual(SEEDS.bm.examples.length, SEEDS.nn.examples.length, "same number of example seeds");

console.log("i18n-parity: OK");
```

- [ ] **Step 2: Extend `test:build` and `test` in `package.json`**

Change `test:build` to also compile `i18n.ts`:
```
"test:build": "tsc src/lib/ml.ts src/lib/corpus.ts src/lib/i18n.ts --rootDir src/lib --outDir test/dist --target ES2020 --module ESNext --moduleResolution bundler --skipLibCheck",
```
Append the parity test to the `test` chain (after `corpus-lang.test.mjs`):
```
... && node test/corpus-lang.test.mjs && node test/i18n-parity.test.mjs && node test/generate-parity.test.mjs ...
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pnpm run test:build`
Expected: FAIL — `tsc` errors that `src/lib/i18n.ts` does not exist (file not found).

- [ ] **Step 4: Create `src/lib/i18n.ts`**

```ts
export type Lang = "bm" | "nn";

export const LANGS: { id: Lang; label: string; htmlLang: string; locale: string }[] = [
  { id: "bm", label: "Bokmål", htmlLang: "nb", locale: "nb-NO" },
  { id: "nn", label: "Nynorsk", htmlLang: "nn", locale: "nn-NO" },
];

export interface Strings {
  header: { title: string; subtitle: string; jump: string };
  hero: {
    badge: string;
    h1Pre: string;
    h1Lang: string; // the gradient language word ("bokmål"/"nynorsk")
    para: string;
    ctaStart: string;
    ctaUnderstand: string;
    stats: { k: string; v: string }[]; // length 3
  };
  understand: {
    title: string;
    intro: string;
    cards: { t: string; d: string; i: string }[]; // length 3
  };
  data: {
    title: string;
    intro: string;
    snippetHeading: string;
    charsTotal: (n: number) => string;
    howHeading: string;
    howPara: (sample: string) => string;
    vocabHeading: (n: number) => string;
  };
  arch: {
    title: string;
    intro: string;
    causalTitle: string;
    causalBody: string;
    headsTitle: string;
    headsBody: string;
    boxInput: { title: string; sub: string };
    boxEmbedding: { title: string; sub: (dim: number) => string };
    boxBlock: { title: (i: number) => string; sub: string };
    boxAttn: { title: string; sub: (heads: number) => string };
    boxFfn: { title: string; sub: string };
    residualNote: string;
    boxFinalNorm: { title: string; sub: string };
    boxOutHead: { title: string; sub: string };
    boxSoftmax: { title: string; sub: string };
    explainHeading: string;
    explain: { b: string; t: string }[]; // length 5
  };
  train: {
    title: string;
    intro: string;
    modelSize: string;
    minibatch: (n: number) => string;
    learningRate: (x: string) => string;
    start: string;
    stop: string;
    reset: string;
    step: (s: number, max: number) => string;
    params: string;
    lossHeading: string;
    lossHelp: string;
    liveLabel: string;
    livePlaceholder: string;
  };
  chat: {
    title: string;
    intro: string;
    promptLabel: string;
    promptPlaceholder: string;
    temp: (x: string) => string;
    tempHelp: string;
    topK: (k: number) => string;
    topKHelp: string;
    length: (n: number) => string;
    lengthHelp: string;
    generate: string;
    thinking: string;
    answerLabel: string;
  };
  rlhf: {
    sectionTitle: string;
    sectionIntro: string;
    introCard: string;
    startBtn: string;
    untrainedHint: string;
    startTextLabel: string;
    creativity: (x: string) => string;
    generatePair: string;
    makingPair: string;
    prefAnswer: (label: string) => string;
    prefBetter: (label: string) => string;
    skip: string;
    trainMore: string;
    stop: string;
    resetTuning: string;
    prefs: string;
    margin: string;
    winRate: string;
    dpoLossHeading: string;
    dpoHelp: string;
  };
  warning: { lead: string; body: string };
  extra: {
    title: string;
    intro: string;
    placeholder: string;
    charsNote: (n: number) => string;
    rebuild: string;
  };
  lossLast: string;
  footer: { line1: string; line2: string };
  docTitle: string;
}

const bm: Strings = {
  header: {
    title: "Språkmodell-trener",
    subtitle: "Lær AI på bokmål – i nettleseren",
    jump: "Hopp til trening →",
  },
  hero: {
    badge: "Ekte trening fra null – ingen ferdig modell",
    h1Pre: "Bygg din egen språkmodell på",
    h1Lang: "bokmål",
    para: "Her trener du en ekte transformator (samme type som ChatGPT) helt fra bunnen av – med ekte baklengs propagasjon og Adam-optimering. Alt skjer lokalt på maskinen din. Følg med steg for steg, og se hvordan tilfeldige tall blir til bokmålstekst.",
    ctaStart: "Start treningen",
    ctaUnderstand: "Forstå hvordan det fungerer",
    stats: [
      { k: "tegn-nivå", v: "tokenisering" },
      { k: "100%", v: "i nettleseren" },
      { k: "fra null", v: "ekte vekter" },
    ],
  },
  understand: {
    title: "Hva er en språkmodell?",
    intro:
      "En språkmodell lærer én enkel ting: å gjette hvilket tegn som kommer neste. Gjør vi det om og om igjen, kan den skrive hele setninger.",
    cards: [
      { t: "Gjett neste tegn", d: "Modellen leser teksten så langt og gjetter hvilken bokstav som bør komme neste.", i: "🔮" },
      { t: "Mål feilen", d: "Vi sammenligner gjettingen med den ekte teksten og regner ut tapet (loss).", i: "📏" },
      { t: "Juster vektene", d: "Backpropagation flytter alle vektene litt mot en bedre gjetting.", i: "🔧" },
    ],
  },
  data: {
    title: "Råtekst og tokenisering",
    intro:
      "Først trenger vi tekst. Her bruker vi norsk bokmål. Datamaskinen forstår ikke bokstaver, så vi deler teksten opp i små enheter – token – og gir hver av dem et tall.",
    snippetHeading: "Utsnitt av treningsdataene (bokmål)",
    charsTotal: (n) => `${n} tegn totalt`,
    howHeading: "Slik blir teksten til tall",
    howPara: (sample) => `Vi deler opp setningen «${sample}» tegn for tegn. Hvert tegn får sin egen ID:`,
    vocabHeading: (n) => `Hele tegnsettet (${n} token = vokabularet)`,
  },
  arch: {
    title: "Modellarkitekturen",
    intro:
      "Vi bruker en transformator – algoritmen bak moderne språkmodeller. Dataene renner oppover gjennom blokkene, og hver blokk lærer noe nytt om sammenhengen i teksten.",
    causalTitle: "Kausal maskering:",
    causalBody:
      "når modellen gjetter posisjon i, får den bare se det som kom før. Slik lærer den å skrive framover, ikke å jukse.",
    headsTitle: 'Flere "hoder":',
    headsBody:
      "multi-head oppmerksomhet lar modellen se på flere ulike ting samtidig – f.eks. både bokstav, ordlyd og betydning.",
    boxInput: { title: "Inndata", sub: "tekst → tegn (token-id-er)" },
    boxEmbedding: { title: "Innbygging (embedding)", sub: (dim) => `tegn + posisjon → ${dim} tall` },
    boxBlock: { title: (i) => `Transformer-blokk ${i}`, sub: "selvoppmerksomhet + feed-forward" },
    boxAttn: { title: "LayerNorm → Multi-head oppmerksomhet", sub: (heads) => `${heads} hoder` },
    boxFfn: { title: "LayerNorm → Feed-forward (GELU)", sub: "ikke-lineær tenkning" },
    residualNote: "+ residual-veier (hopp over ledd)",
    boxFinalNorm: { title: "Sluttnormalisering", sub: "LayerNorm" },
    boxOutHead: { title: "Utgangshode", sub: "→ poengsum (logits) for hvert tegn" },
    boxSoftmax: { title: "Softmax", sub: "→ sannsynlighet for hvilket tegn som kommer neste" },
    explainHeading: "Hva skjer inni?",
    explain: [
      { b: "Innbygging:", t: "hvert tegn blir til en liste med tall, og vi legger til informasjon om hvor i teksten det står." },
      { b: "Selvoppmerksomhet:", t: "hvert tegn ser på de andre tegnene og finner ut hva som er viktig i sammenhengen." },
      { b: "Feed-forward:", t: "et lite nevralt nett som «tenker» videre over hver posisjon." },
      { b: "Residualveier:", t: "informasjonen hopper over hvert ledd slik at ingenting går tapt." },
      { b: "Softmax:", t: "gjør poengene om til sannsynlighet – slik velger modellen neste tegn." },
    ],
  },
  train: {
    title: "Trening – se modellen lære",
    intro:
      "Nå setter vi i gang. For hvert steg gjetter modellen, måler tapet, og flytter vektene med Adam-optimering. Se om tapet går ned – da skjer læringen!",
    modelSize: "Modellstørrelse",
    minibatch: (n) => `Minibatch: ${n}`,
    learningRate: (x) => `Læringsrate: ${x}`,
    start: "▶ Start trening",
    stop: "⏸ Stopp",
    reset: "↺ Nullstill",
    step: (s, max) => `Steg ${s} / ${max}`,
    params: "parametere",
    lossHeading: "Tap (loss) over tid",
    lossHelp:
      "Lavere tap = bedre. En perfekt modell ville hatt tap rundt 0. Jo raskere kurven søker nedover, jo fortere lærer modellen.",
    liveLabel: "Dette skriver modellen nå",
    livePlaceholder: "Trykk «Start trening» for å se eksempler underveis…",
  },
  chat: {
    title: "Prøv modellen",
    intro:
      "Skriv en starttekst, og la modellen fortsette. Den gjetter ett tegn om gangen. Små modeller gir ikke perfekte svar – men se hvor mye bedre det blir etter hvert som den trener!",
    promptLabel: "Din starttekst (bokmål)",
    promptPlaceholder: "f.eks. «Det var en gang»",
    temp: (x) => `Temperatur: ${x}`,
    tempHelp: "0 = trygg, høy = kreativ",
    topK: (k) => `Top-k: ${k}`,
    topKHelp: "bare de k beste valgene",
    length: (n) => `Lengde: ${n} tegn`,
    lengthHelp: "hvor mange nye tegn",
    generate: "✨ Generer tekst",
    thinking: "Tenker…",
    answerLabel: "Svar fra modellen",
  },
  rlhf: {
    sectionTitle: "RLHF – lær modellen hva vi foretrekker",
    sectionIntro:
      "Etter grunntreningen kan vi finjustere modellen med menneskelig tilbakemelding. Du velger hvilken av to fortsettelser som er best, og modellen blir dyttet mot valget ditt med DPO – forankret til en frossen kopi av modellen.",
    introCard:
      "RLHF («Reinforcement Learning from Human Feedback») lærer modellen hva slags svar vi mennesker foretrekker. Vi viser deg to fortsettelser, du velger den beste, og modellen blir justert mot valget ditt – forankret til en frossen referansemodell (DPO).",
    startBtn: "Start preferansetrening",
    untrainedHint: "Tips: tren modellen først i steg 3 – da blir fortsettelsene mer meningsfulle.",
    startTextLabel: "Starttekst",
    creativity: (x) => `Kreativitet: ${x}`,
    generatePair: "↻ Generer et par",
    makingPair: "Lager par…",
    prefAnswer: (label) => `Svar ${label}`,
    prefBetter: (label) => `👍 ${label} er bedre`,
    skip: "Hopp over (likeverdige)",
    trainMore: "Tren mer på preferansene",
    stop: "⏸ Stopp",
    resetTuning: "↺ Nullstill justering",
    prefs: "Preferanser:",
    margin: "Margin:",
    winRate: "Vinner-rate:",
    dpoLossHeading: "DPO-tap over tid",
    dpoHelp:
      "Margin = hvor mye mer sannsynlig den valgte fortsettelsen er enn den avviste, sammenlignet med referansemodellen. Høyere margin og vinner-rate = modellen følger preferansene dine.",
  },
  warning: {
    lead: "Advarsel – ærlig om hva dette er:",
    body:
      " Dette er en svært liten modell som blir trent i nettleseren din på noen få setninger. Den kan ikke måle seg med store modeller som ChatGPT, som er titusenvis av ganger større og trener i uker på enorme mengder data. Men prinsippet er nøyaktig det samme: ekte transformator, ekte backpropagation, ekte læring. Mer tekst og flere steg gir bedre resultat – prøv å lime inn egen tekst i feltet under!",
  },
  extra: {
    title: "Legg til egen tekst",
    intro:
      "Mer og variert tekst gjør modellen bedre. Lim inn bokmålstekst her (f.eks. fra en bok eller noe du har skrevet). Modellen blir bygd på nytt med de nye dataene.",
    placeholder: "Lim inn bokmålstekst her… (gjerne flere avsnitt)",
    charsNote: (n) => `Tatt med i tillegg til ${n} faste tegn.`,
    rebuild: "Bygg modell på nytt",
  },
  lossLast: "Siste tap:",
  footer: {
    line1:
      "Bygd med egen skrevet maskinlæringsmotor – transformator, autograd og Adam – helt i JavaScript.",
    line2: "All kode og all læring skjer lokalt i din egen nettleser. 🇳🇴",
  },
  docTitle: "Språkmodell-trener – bygg AI på bokmål",
};

const nn: Strings = {
  header: {
    title: "Språkmodell-trener",
    subtitle: "Lær AI på nynorsk – i nettlesaren",
    jump: "Hopp til trening →",
  },
  hero: {
    badge: "Ekte trening frå null – ingen ferdig modell",
    h1Pre: "Bygg din eigen språkmodell på",
    h1Lang: "nynorsk",
    para: "Her trenar du ein ekte transformator (samme type som ChatGPT) heilt frå bunnen av – med ekte baklengs propagasjon og Adam-optimering. Alt skjer lokalt i maskina di. Følg med steg for steg, og sjå korleis tilfeldige tal blir til nynorsk tekst.",
    ctaStart: "Start treninga",
    ctaUnderstand: "Forstå korleis det fungerer",
    stats: [
      { k: "teikn-nivå", v: "tokenisering" },
      { k: "100%", v: "i nettlesaren" },
      { k: "frå null", v: "ekte vektar" },
    ],
  },
  understand: {
    title: "Kva er ei språkmodell?",
    intro:
      "Ei språkmodell lærer éin enkel ting: å gjetta kva teikn som kjem neste. Gjer vi det om og om igjen, kan ho skriva heile setningar.",
    cards: [
      { t: "Gjet neste teikn", d: "Modellen les teksten så langt og gjet kva bokstav som bør koma neste.", i: "🔮" },
      { t: "Mål feilen", d: "Vi samanliknar gjettinga med den ekte teksten og rekna ut tapet (loss).", i: "📏" },
      { t: "Juster vektane", d: "Backpropagation flyttar alle vektane litt mot ei betre gjetting.", i: "🔧" },
    ],
  },
  data: {
    title: "Råtekst og tokenisering",
    intro:
      "Først treng vi tekst. Her bruker vi norsk nynorsk. Datamaskina forstår ikkje bokstavar, så vi deler teksten opp i små einingar – token – og gir kvar av dei eit tal.",
    snippetHeading: "Utsnitt av treningsdataa (nynorsk)",
    charsTotal: (n) => `${n} teikn totalt`,
    howHeading: "Slik blir teksten til tal",
    howPara: (sample) => `Vi delar opp setninga «${sample}» teikn for teikn. Kvart teikn får sin eigen ID:`,
    vocabHeading: (n) => `Heile teiknsettet (${n} token = vokabularet)`,
  },
  arch: {
    title: "Modellarkitekturen",
    intro:
      "Vi nyttar ein transformator – algoritmen bak moderne språkmodellar. Dataen renn oppover gjennom blokkane, og kvar blokk lærer noko nytt om samanhengen i teksten.",
    causalTitle: "Kausal maskering:",
    causalBody:
      "når modellen gjet posisjon i, får ho berre sjå det som kom før. Slik lærer ho å skriva framover, ikkje å juksa.",
    headsTitle: 'Fleire "hovud":',
    headsBody:
      "multi-head oppmerksomheit let modellen sjå på fleire ulike ting samtidig – t.d. både bokstav, ordlyd og tyding.",
    boxInput: { title: "Inndata", sub: "tekst → teikn (token-id-ar)" },
    boxEmbedding: { title: "Innbygging (embedding)", sub: (dim) => `teikn + posisjon → ${dim} tal` },
    boxBlock: { title: (i) => `Transformer-blokk ${i}`, sub: "sjølvoppmerksomhet + feed-forward" },
    boxAttn: { title: "LayerNorm → Multi-head oppmerksomheit", sub: (heads) => `${heads} hovud` },
    boxFfn: { title: "LayerNorm → Feed-forward (GELU)", sub: "ikkje-lineær tenking" },
    residualNote: "+ residual-vegar (sprang over ledd)",
    boxFinalNorm: { title: "Slutt-normalisering", sub: "LayerNorm" },
    boxOutHead: { title: "Utgangshovud", sub: "→ poengsum (logits) for kvart teikn" },
    boxSoftmax: { title: "Softmax", sub: "→ sannsyn for kva teikn som kjem neste" },
    explainHeading: "Kva skjer inni?",
    explain: [
      { b: "Innbygging:", t: "kvart teikn blir til ei liste med tal, og vi legg til informasjon om kvar i teksten det står." },
      { b: "Sjølvoppmerksomheit:", t: "kvart teikn ser på dei andre teikna og finn ut kva som er viktig i samanhengen." },
      { b: "Feed-forward:", t: "eit lite nevralt nett som «tenkjer» vidare over kvar posisjon." },
      { b: "Residualvegar:", t: "informasjonen hoppar over kvart ledd slik at ingenting går tapt." },
      { b: "Softmax:", t: "gjer poenga om til sannsyn – slik vel modellen neste teikn." },
    ],
  },
  train: {
    title: "Trening – sjå modellen læra",
    intro:
      "No set vi i gong. For kvart steg gjet modellen, måler tapet, og flyttar vektane med Adam-optimering. Sjå om tapet går ned – då skjer læringa!",
    modelSize: "Modellstørrelse",
    minibatch: (n) => `Minibatch: ${n}`,
    learningRate: (x) => `Læringsrate: ${x}`,
    start: "▶ Start trening",
    stop: "⏸ Stopp",
    reset: "↺ Nullstill",
    step: (s, max) => `Steg ${s} / ${max}`,
    params: "parametrar",
    lossHeading: "Tap (loss) over tid",
    lossHelp:
      "Lågare tap = betre. Ein perfekt modell ville hatt tap rundt 0. Jo raskare kurva søkjer nedover, jo fortare lærer modellen.",
    liveLabel: "Dette skriv modellen no",
    livePlaceholder: "Trykk «Start trening» for å sjå døme undervegs…",
  },
  chat: {
    title: "Prøv modellen",
    intro:
      "Skriv ein starttekst, og lat modellen halda fram. Ho gjet eitt teikn om gongen. Små modellar gir ikkje perfekte svar – men sjå kor mykje betre det blir etter kvart som ho trenar!",
    promptLabel: "Din starttekst (nynorsk)",
    promptPlaceholder: "t.d. «Det var ein gong»",
    temp: (x) => `Temperatur: ${x}`,
    tempHelp: "0 = trygg, høg = kreativ",
    topK: (k) => `Top-k: ${k}`,
    topKHelp: "berre dei k beste vala",
    length: (n) => `Lengd: ${n} teikn`,
    lengthHelp: "kor mange nye teikn",
    generate: "✨ Generer tekst",
    thinking: "Tenkjer…",
    answerLabel: "Svar frå modellen",
  },
  rlhf: {
    sectionTitle: "RLHF – lær modellen kva vi føretrekkjer",
    sectionIntro:
      "Etter grunntreninga kan vi finjustere modellen med menneskeleg tilbakemelding. Du vel kva for eit av to framhald som er best, og modellen blir dytta mot valet ditt med DPO – forankra til ein frosen kopi av modellen.",
    introCard:
      "RLHF («Reinforcement Learning from Human Feedback») lærer modellen kva slags svar vi menneske føretrekkjer. Vi viser deg to framhald, du vel det beste, og modellen blir justert mot valet ditt – forankra til ein frosen referansemodell (DPO).",
    startBtn: "Start preferanse-trening",
    untrainedHint: "Tips: tren modellen først i steg 3 – då blir framhalda meir meiningsfulle.",
    startTextLabel: "Starttekst",
    creativity: (x) => `Kreativitet: ${x}`,
    generatePair: "↻ Generer eit par",
    makingPair: "Lagar par…",
    prefAnswer: (label) => `Svar ${label}`,
    prefBetter: (label) => `👍 ${label} er betre`,
    skip: "Hopp over (likeverdige)",
    trainMore: "Tren meir på preferansane",
    stop: "⏸ Stopp",
    resetTuning: "↺ Nullstill justering",
    prefs: "Preferansar:",
    margin: "Margin:",
    winRate: "Vinnar-rate:",
    dpoLossHeading: "DPO-tap over tid",
    dpoHelp:
      "Margin = kor mykje meir sannsynleg det valde framhaldet er enn det avviste, samanlikna med referansemodellen. Høgare margin og vinnar-rate = modellen følgjer preferansane dine.",
  },
  warning: {
    lead: "Åtvaring – ærlig om kva dette er:",
    body:
      " Dette er ein svært liten modell som blir trent i nettlesaren din på nokre få setningar. Ho kan ikkje måla seg med store modellar som ChatGPT, som er titusenvis av gonger større og trenar i veker på enorme mengder data. Men prinsippet er nøyaktig det same: ekte transformator, ekte backpropagation, ekte læring. Meir tekst og fleire steg gir betre resultat – prøv å lime inn eigen tekst i feltet under!",
  },
  extra: {
    title: "Legg til eigen tekst",
    intro:
      "Meir og variert tekst gjer modellen betre. Lim inn nynorsk tekst her (t.d. frå ei bok eller noko du har skrive). Modellen blir bygd på nytt med den nye dataa.",
    placeholder: "Lim inn nynorsk tekst her… (gjerne fleire avsnitt)",
    charsNote: (n) => `Teken med i tillegg til ${n} faste teikn.`,
    rebuild: "Bygg modell på nytt",
  },
  lossLast: "Siste tap:",
  footer: {
    line1:
      "Bygt med eigen skreve maskinlæringsmotor – transformator, autograd og Adam – heilt i JavaScript.",
    line2: "All kode og all læring skjer lokalt i din eigen nettlesar. 🇳🇴",
  },
  docTitle: "Språkmodell-trener – bygg AI på nynorsk",
};

export const STRINGS: Record<Lang, Strings> = { bm, nn };

export interface Seeds {
  chatPrompt: string;
  examples: string[];
  sampleSentence: string;
  trainSeed: string;
}

export const SEEDS: Record<Lang, Seeds> = {
  bm: {
    chatPrompt: "Det var en gang",
    examples: ["Det var en gang", "Norge er", "Jeg heter", "Vann er"],
    sampleSentence: "Norge er et land",
    trainSeed: "Det var en gang",
  },
  nn: {
    chatPrompt: "Det var ein gong",
    examples: ["Det var ein gong", "Noreg er", "Eg heiter", "Vatn er"],
    sampleSentence: "Noreg er eit land",
    trainSeed: "Det var ein gong",
  },
};
```

- [ ] **Step 5: Run the parity test to verify it passes**

Run: `pnpm run test:build && node test/i18n-parity.test.mjs`
Expected: `i18n-parity: OK`. (If `deepEqual` fails, a key is missing/mismatched between `bm` and `nn` — fix until shapes match.)

- [ ] **Step 6: Run the full suite**

Run: `pnpm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/lib/i18n.ts test/i18n-parity.test.mjs package.json
git commit -m "feat(i18n): bilingual string bundles and seeds with parity test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire the UI to i18n (LossChart, Architecture, Rlhf, App, index.html)

This is one integration task: the `s: Strings` prop contract spans every component, so the project only typechecks once all are converted. Do every edit, then verify with `pnpm typecheck` + `pnpm build` + `pnpm test`, then commit once. There is no automated UI test; the gate is a clean typecheck/build, green unit tests, and the manual smoke checklist in Task 4.

**Files:**
- Modify: `src/components/LossChart.tsx` (add `label` prop)
- Modify: `src/components/Architecture.tsx` (add `s: Strings` prop; replace literals)
- Modify: `src/components/Rlhf.tsx` (add `s: Strings` prop; replace literals)
- Modify: `src/App.tsx` (lang state, toggle, resolve & pass `s`/seeds, rebuild wiring, effects)
- Modify: `index.html` (static Bokmål default)

**Interfaces:**
- Consumes: `STRINGS`, `SEEDS`, `LANGS`, `Lang`, `Strings` from `@/lib/i18n`; `corpora` from `@/lib/corpus`.
- Produces: `LossChart` accepts `label?: string`; `Architecture` and `Rlhf` accept `s: Strings`.

- [ ] **Step 1: `LossChart.tsx` — accept a `label` prop**

Open `src/components/LossChart.tsx`. Find the component's prop type and signature (currently `data`-only) and the JSX line rendering `Siste tap:` (around line 71):

```tsx
<span>Siste tap: <span className="font-semibold text-indigo-600">{last.toFixed(4)}</span></span>
```

Add an optional `label` prop (default keeps current text) and use it:
- In the props type, add `label?: string;`.
- In the destructure, add `label = "Siste tap:"`.
- Change the line to:
```tsx
<span>{label} <span className="font-semibold text-indigo-600">{last.toFixed(4)}</span></span>
```

- [ ] **Step 2: `Architecture.tsx` — take `s: Strings`, replace all literals**

At the top add: `import type { Strings } from "@/lib/i18n";`

Change `interface Props { layers; heads; dim }` to also include `s: Strings`, and the signature to `export default function Architecture({ layers, heads, dim, s }: Props)`.

Replace the hardcoded Norwegian in the JSX with `s.arch.*` (the existing Nynorsk literals now live in the bundle). Apply exactly these substitutions:

| Current literal in JSX | Replace with |
|---|---|
| `title="Inndata" sub="tekst → teikn (token-id-ar)"` | `title={s.arch.boxInput.title} sub={s.arch.boxInput.sub}` |
| `title="Innbygging (embedding)" sub={`teikn + posisjon → ${dim} tal`}` | `title={s.arch.boxEmbedding.title} sub={s.arch.boxEmbedding.sub(dim)}` |
| `title={`Transformer-blokk ${i + 1}`} sub="sjølvoppmerksomhet + feed-forward"` | `title={s.arch.boxBlock.title(i + 1)} sub={s.arch.boxBlock.sub}` |
| `title="LayerNorm → Multi-head oppmerksomheit" sub={`${heads} hovud`}` | `title={s.arch.boxAttn.title} sub={s.arch.boxAttn.sub(heads)}` |
| `title="LayerNorm → Feed-forward (GELU)" sub="ikkje-lineær tenking"` | `title={s.arch.boxFfn.title} sub={s.arch.boxFfn.sub}` |
| `+ residual-vegar (sprang over ledd)` (text node) | `{s.arch.residualNote}` |
| `title="Slutt-normalisering" sub="LayerNorm"` | `title={s.arch.boxFinalNorm.title} sub={s.arch.boxFinalNorm.sub}` |
| `title="Utgangshovud" sub="→ poengsum (logits) for kvart teikn"` | `title={s.arch.boxOutHead.title} sub={s.arch.boxOutHead.sub}` |
| `title="Softmax" sub="→ sannsyn for kva teikn som kjem neste"` | `title={s.arch.boxSoftmax.title} sub={s.arch.boxSoftmax.sub}` |
| `<h4 ...>Kva skjer inni?</h4>` | `<h4 ...>{s.arch.explainHeading}</h4>` |

Replace the 5 `<li>` explanation items with a map over `s.arch.explain` (keeps colors via index; the original used 3 distinct text colors — preserve them by index):

```tsx
{s.arch.explain.map((e, i) => {
  const colors = ["text-indigo-600", "text-violet-600", "text-violet-600", "text-emerald-600", "text-amber-600"];
  return (
    <li key={i}>
      <b className={colors[i]}>{e.b}</b> {e.t}
    </li>
  );
})}
```

- [ ] **Step 3: `Rlhf.tsx` — take `s: Strings`, replace all literals**

At the top add: `import type { Strings } from "@/lib/i18n";`

Change the default export signature to:
```tsx
export default function Rlhf({ rlhf, examples, s }: { rlhf: RlhfApi; examples: string[]; s: Strings }) {
```

`PrefCard` needs two strings; pass them in. Change `PrefCard`'s props to add `answerLabel: string; betterLabel: string;` and use them:
- `<div ...>Svar {label}</div>` → `<div ...>{answerLabel}</div>`
- `👍 {label} er betre` → `{betterLabel}`

At the `PrefCard` call sites add the new props:
```tsx
<PrefCard label="A" answerLabel={s.rlhf.prefAnswer("A")} betterLabel={s.rlhf.prefBetter("A")} text={rlhf.pairA?.text ?? ""} onPick={() => rlhf.choose("A")} disabled={busy} />
<PrefCard label="B" answerLabel={s.rlhf.prefAnswer("B")} betterLabel={s.rlhf.prefBetter("B")} text={rlhf.pairB?.text ?? ""} onPick={() => rlhf.choose("B")} disabled={busy} />
```

Apply these substitutions in `Rlhf`'s JSX:

| Current literal | Replace with |
|---|---|
| intro `<p>RLHF («Reinforcement Learning…företrekkjer (DPO).</p>` text | `{s.rlhf.introCard}` |
| `Start preferanse-trening` | `{s.rlhf.startBtn}` |
| `Tips: tren modellen først i steg 3 … meiningsfulle.` | `{s.rlhf.untrainedHint}` |
| `<label ...>Starttekst</label>` | `<label ...>{s.rlhf.startTextLabel}</label>` |
| `Kreativitet: {rlhf.temp.toFixed(2)}` | `{s.rlhf.creativity(rlhf.temp.toFixed(2))}` |
| `{rlhf.generating ? "Lagar par…" : "↻ Generer eit par"}` | `{rlhf.generating ? s.rlhf.makingPair : s.rlhf.generatePair}` |
| `Hopp over (likeverdige)` | `{s.rlhf.skip}` |
| `Tren meir på preferansane` | `{s.rlhf.trainMore}` |
| `⏸ Stopp` | `{s.rlhf.stop}` |
| `↺ Nullstill justering` | `{s.rlhf.resetTuning}` |
| `Preferansar:` | `{s.rlhf.prefs}` |
| `Margin:` | `{s.rlhf.margin}` |
| `Vinnar-rate:` | `{s.rlhf.winRate}` |
| `<h3 ...>DPO-tap over tid</h3>` | `<h3 ...>{s.rlhf.dpoLossHeading}</h3>` |
| `Margin = kor mykje meir sannsynleg … preferansane dine.` | `{s.rlhf.dpoHelp}` |
| `<LossChart data={rlhf.losses} />` | `<LossChart data={rlhf.losses} label={s.lossLast} />` |

- [ ] **Step 4: `App.tsx` — imports, language state, helpers**

Update the imports at the top:
```tsx
import { buildTokenizer, corpora } from "@/lib/corpus";
import { STRINGS, SEEDS, LANGS, type Lang } from "@/lib/i18n";
```
(Remove the old `corpus` import; it is replaced by `corpora`/`activeCorpus`.)

Add this helper above `export default function App()`:
```tsx
const LANG_KEY = "trainer-lang";
function readStoredLang(): Lang {
  try {
    const v = localStorage.getItem(LANG_KEY);
    if (v === "bm" || v === "nn") return v;
  } catch {
    /* localStorage unavailable */
  }
  return "bm";
}
function writeStoredLang(l: Lang) {
  try {
    localStorage.setItem(LANG_KEY, l);
  } catch {
    /* ignore */
  }
}
```

At the very top of `App()` (with the other config state), add:
```tsx
const [lang, setLang] = useState<Lang>(() => readStoredLang());
const s = STRINGS[lang];
const seed = SEEDS[lang];
const activeCorpus = corpora[lang];
const activeLocale = LANGS.find((l) => l.id === lang)!.locale;
```

- [ ] **Step 5: `App.tsx` — use the active corpus and seeds in logic**

- In `buildEngine`, change `const fullText = corpus + "\n" + customText;` to `const fullText = activeCorpus + "\n" + customText;` and add `activeCorpus` to its `useCallback` dependency array (alongside `preset`, `rlhf.reset`). This makes a language change rebuild the engine via the existing `useEffect(..., [buildEngine])`.
- In the training `loop`, change the hardcoded sampler seed `"Det var ein gong"` to `seed.trainSeed`. Because `loop` is a `useCallback`, add `seed.trainSeed` to its dependency array (next to `cfg.batch`).
- Replace `const [chatPrompt, setChatPrompt] = useState("Det var ein gong");` with `const [chatPrompt, setChatPrompt] = useState(seed.chatPrompt);`
- Replace `const sampleSentence = "Noreg er eit land";` with `const sampleSentence = seed.sampleSentence;`
- Replace `const examples = ["Det var ein gong", "Noreg er", "Eg heiter", "Vatn er"];` with `const examples = seed.examples;`
- Change `displayTok` memo from `buildTokenizer(corpus)` to `buildTokenizer(activeCorpus)` with dep `[activeCorpus]`; update `sampleTokens` and `vocabList` memos to depend on `displayTok` (already do) and `sampleSentence`.
- In `stats`, replace the two `corpus.length`/`corpus` fallbacks with `activeCorpus` (`chars: eng?.data.length ?? activeCorpus.length`), and add `activeCorpus` to the memo deps.
- Replace `stats.params.toLocaleString("nn")` with `stats.params.toLocaleString(activeLocale)`.
- In the "egen tekst" section, replace `{corpus.length} faste teikn` usage — it is produced by `s.extra.charsNote(activeCorpus.length)` (see Step 7 table).

- [ ] **Step 6: `App.tsx` — language effects (persist, document, seed reset)**

Add these effects near the other `useEffect`s:
```tsx
// Persist + reflect language on <html> and title.
useEffect(() => {
  writeStoredLang(lang);
  const meta = LANGS.find((l) => l.id === lang)!;
  document.documentElement.lang = meta.htmlLang;
  document.title = s.docTitle;
}, [lang, s.docTitle]);

// On language change, repopulate the editable prompts with the new seed.
useEffect(() => {
  setChatPrompt(seed.chatPrompt);
  rlhf.setPrompt(seed.chatPrompt);
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [lang]);
```

- [ ] **Step 7: `App.tsx` — replace all UI literals and add the header toggle**

Add the toggle in the header, right after the `subtitle` `<div>` block (inside the left `flex items-center gap-2` container, or as a sibling before the "Hopp til trening" link). Use a compact segmented control driven by `LANGS`:
```tsx
<div className="ml-3 inline-flex overflow-hidden rounded-lg border border-slate-300 text-xs font-semibold">
  {LANGS.map((l) => (
    <button
      key={l.id}
      onClick={() => setLang(l.id)}
      disabled={running || rlhf.dpoRunning}
      className={cn(
        "px-2.5 py-1 transition disabled:opacity-50",
        lang === l.id ? "bg-slate-900 text-white" : "bg-white text-slate-600 hover:bg-slate-50"
      )}
    >
      {l.label}
    </button>
  ))}
</div>
```

Then apply these literal → expression substitutions throughout `App.tsx` (each left value is the exact current text; replace with the right expression):

| Current literal | Replace with |
|---|---|
| `Lær AI på nynorsk – i nettlesaren` | `{s.header.subtitle}` |
| `Hopp til trening →` (header link) | `{s.header.jump}` |
| `Ekte trening frå null – ingen ferdig modell` | `{s.hero.badge}` |
| `Bygg din eigen språkmodell på` (+ `{" "}`) | `{s.hero.h1Pre}{" "}` |
| `nynorsk` (gradient `<span>`) | `{s.hero.h1Lang}` |
| hero `<p>Her trenar du … nynorsk tekst.</p>` body | `{s.hero.para}` |
| `Start treninga` | `{s.hero.ctaStart}` |
| `Forstå korleis det fungerer` | `{s.hero.ctaUnderstand}` |
| hero stats array `[{k:"teikn-nivå",…},…]` | `{s.hero.stats}` (map over it instead of the inline array) |
| Section "Forstå" `title="Kva er ei språkmodell?"` | `title={s.understand.title}` |
| its `intro="Ei språkmodell lærer…"` | `intro={s.understand.intro}` |
| the 3 understand cards inline array | `s.understand.cards` (map over it; keep `Steg {s.n}` → `Steg {String(i + 1)}`) |
| `Steg {s.n}` label text | `Steg {i + 1}` (index-based, since cards no longer carry `n`) |
| Section "data" `title="Råtekst og tokenisering"` | `title={s.data.title}` |
| its `intro="Først treng vi tekst…"` | `intro={s.data.intro}` |
| `Utsnitt av treningsdataa (nynorsk)` | `{s.data.snippetHeading}` |
| `{stats.chars} teikn totalt` | `{s.data.charsTotal(stats.chars)}` |
| `{corpus.slice(0, 420)}…` | `{activeCorpus.slice(0, 420)}…` |
| `Slik blir teksten til tal` | `{s.data.howHeading}` |
| `Vi delar opp setninga «{sampleSentence}» teikn for teikn. Kvart teikn får sin eigen ID:` | `{s.data.howPara(sampleSentence)}` |
| `Heile teiknsettet ({vocabList.length} token = vokabularet)` | `{s.data.vocabHeading(vocabList.length)}` |
| Section "arkitektur" `title="Modellarkitekturen"` | `title={s.arch.title}` |
| its `intro="Vi nyttar ein transformator…"` | `intro={s.arch.intro}` |
| `<b ...>Kausal maskering:</b> når modellen gjet…juksa.` | `<b className="text-slate-800">{s.arch.causalTitle}</b> {s.arch.causalBody}` |
| `<b ...>Fleire "hovud":</b> multi-head…tyding.` | `<b className="text-slate-800">{s.arch.headsTitle}</b> {s.arch.headsBody}` |
| `<Architecture layers={cfg.nLayer} heads={cfg.nHead} dim={cfg.dim} />` | add `s={s}`: `<Architecture layers={cfg.nLayer} heads={cfg.nHead} dim={cfg.dim} s={s} />` |
| Section "trening" `title="Trening – sjå modellen læra"` | `title={s.train.title}` |
| its `intro="No set vi i gong…"` | `intro={s.train.intro}` |
| `Modellstørrelse` | `{s.train.modelSize}` |
| `Minibatch: {cfg.batch}` | `{s.train.minibatch(cfg.batch)}` |
| `Læringsrate: {lr.toFixed(4)}` | `{s.train.learningRate(lr.toFixed(4))}` |
| `▶ Start trening` | `{s.train.start}` |
| `⏸ Stopp` (training) | `{s.train.stop}` |
| `↺ Nullstill` | `{s.train.reset}` |
| `Steg {step} / {MAX_STEPS}` | `{s.train.step(step, MAX_STEPS)}` |
| `{stats.params.toLocaleString("nn")} parametrar` | `{stats.params.toLocaleString(activeLocale)} {s.train.params}` |
| `Tap (loss) over tid` | `{s.train.lossHeading}` |
| `<LossChart data={losses} />` | `<LossChart data={losses} label={s.lossLast} />` |
| `Lågare tap = betre. … lærer modellen.` | `{s.train.lossHelp}` |
| `Dette skriv modellen no` | `{s.train.liveLabel}` |
| `Trykk «Start trening» for å sjå døme undervegs…` | `{s.train.livePlaceholder}` |
| Section "chat" `title="Prøv modellen"` | `title={s.chat.title}` |
| its `intro="Skriv ein starttekst…"` | `intro={s.chat.intro}` |
| `Din starttekst (nynorsk)` | `{s.chat.promptLabel}` |
| `placeholder="t.d. «Det var ein gong»"` | `placeholder={s.chat.promptPlaceholder}` |
| `Temperatur: {genTemp.toFixed(2)}` | `{s.chat.temp(genTemp.toFixed(2))}` |
| `0 = trygg, høg = kreativ` | `{s.chat.tempHelp}` |
| `Top-k: {genTopK}` | `{s.chat.topK(genTopK)}` |
| `berre dei k beste vala` | `{s.chat.topKHelp}` |
| `Lengd: {genLen} teikn` | `{s.chat.length(genLen)}` |
| `kor mange nye teikn` | `{s.chat.lengthHelp}` |
| `{genLoading ? "Tenkjer…" : "✨ Generer tekst"}` | `{genLoading ? s.chat.thinking : s.chat.generate}` |
| `Svar frå modellen` | `{s.chat.answerLabel}` |
| Section "rlhf" `title="RLHF – lær modellen kva vi føretrekkjer"` | `title={s.rlhf.sectionTitle}` |
| its `intro="Etter grunntreninga…"` | `intro={s.rlhf.sectionIntro}` |
| `<Rlhf rlhf={rlhf} examples={examples} />` | `<Rlhf rlhf={rlhf} examples={examples} s={s} />` |
| `<b>Åtvaring – ærlig om kva dette er:</b>` + body text | `<b>{s.warning.lead}</b>{s.warning.body}` |
| Section "eigentekst" `title="Legg til eigen tekst"` | `title={s.extra.title}` |
| its `intro="Meir og variert tekst…"` | `intro={s.extra.intro}` |
| `placeholder="Lim inn nynorsk tekst her… (gjerne fleire avsnitt)"` | `placeholder={s.extra.placeholder}` |
| `Teken med i tillegg til {corpus.length} faste teikn.` | `{s.extra.charsNote(activeCorpus.length)}` |
| `Bygg modell på nytt` | `{s.extra.rebuild}` |
| footer `Bygt med eigen skreve maskinlæringsmotor … JavaScript.` | `{s.footer.line1}` |
| footer `All kode og all læring skjer lokalt i din eigen nettlesar. 🇳🇴` | `{s.footer.line2}` |

For the hero stats / understand cards that were inline arrays, map over the bundle arrays instead. Example for stats:
```tsx
{s.hero.stats.map((st) => (
  <div key={st.v} className="rounded-xl border border-slate-200 bg-white/70 px-3 py-3">
    <div className="text-lg font-bold text-indigo-600">{st.k}</div>
    <div className="text-xs text-slate-500">{st.v}</div>
  </div>
))}
```
And for understand cards, map `s.understand.cards` with index `i`, using `Steg {i + 1}`, `{c.t}`, `{c.d}`, `{c.i}`.

- [ ] **Step 8: `index.html` — Bokmål static default**

```html
<!doctype html>
<html lang="nb">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Språkmodell-trener – bygg AI på bokmål</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 9: Typecheck**

Run: `pnpm typecheck`
Expected: no errors. (Common misses: a literal left unconverted, a missing `s` prop on `Architecture`/`Rlhf`, or a function-string called without its argument.)

- [ ] **Step 10: Build**

Run: `pnpm build`
Expected: Vite build succeeds, single-file bundle emitted to `dist/`.

- [ ] **Step 11: Run the unit suite**

Run: `pnpm test`
Expected: all green.

- [ ] **Step 12: Commit**

```bash
git add src/components/LossChart.tsx src/components/Architecture.tsx src/components/Rlhf.tsx src/App.tsx index.html
git commit -m "feat(i18n): bilingual UI with Bokmål default and language toggle

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Manual verification + branch finish

**Files:** none (verification only).

- [ ] **Step 1: Run the dev server and smoke-test**

Run: `pnpm dev` and open the printed URL.

Verify:
- First load (clear `localStorage` first via devtools, key `trainer-lang`) shows a **fully Bokmål** UI: header subtitle "Lær AI på bokmål…", hero word "bokmål", every section, the Architecture diagram boxes, the RLHF panel, and footer all in Bokmål. The corpus snippet is the Bokmål text.
- Click **Nynorsk** in the header toggle: the entire UI flips to Nynorsk, the corpus snippet becomes the Nynorsk text, the chat prompt/examples/sample sentence switch to Nynorsk seeds, and training resets (step 0, empty loss chart).
- Start training a few steps in each language; confirm the live sample reflects the active corpus and nothing throws.
- The toggle is **disabled** while training/DPO is running.
- Reload the page: the last-selected language is restored; `document.documentElement.lang` and the tab title match it (inspect in devtools).

- [ ] **Step 2: Final full check**

Run: `pnpm typecheck && pnpm test && pnpm build`
Expected: all pass.

- [ ] **Step 3: Decide integration**

Per `superpowers:finishing-a-development-branch`: present merge / PR / cleanup options for branch `feat/bilingual-bokmaal-nynorsk`.

---

## Self-Review (completed during planning)

- **Spec coverage:** full bilingual UI (Task 2 bundles + Task 3 wiring), Bokmål default (Task 1 `corpus = corpora.bm`, Task 3 `useState(readStoredLang)` → `"bm"`, index.html), fresh Bokmål corpus (Task 1), header toggle disabled mid-training (Task 3 Step 7), seed swap + reset (Task 3 Steps 5–6), engine rebuild on switch (Task 3 Step 5), localStorage persistence (Task 3 Steps 4/6), document lang/title sync (Task 3 Step 6), back-compat tests + parity test (Tasks 1–2). All acceptance criteria map to a task.
- **Placeholder scan:** no TBD/TODO; every string literal is provided in full in the bundles; component edits are exact literal→expression substitution tables.
- **Type consistency:** `Strings`/`Seeds`/`Lang` names used identically across i18n.ts, App, Architecture, Rlhf; `corpora` map keys `bm`/`nn` consistent; `LossChart` `label` prop consistent across its two call sites.
