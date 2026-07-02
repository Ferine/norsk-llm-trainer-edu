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
    charTooltip: (i: number) => string;
  };
  bpe: {
    title: string;
    intro: string;
    mergeBtn: string;
    resetBtn: string;
    mergeCount: (k: number, n: number) => string;
    thisMergeHeading: string;
    noMergeYet: string;
    foundTimes: (n: number) => string;
    rivalsLabel: string;
    rulesHeading: string;
    noRules: string;
    sentenceHeading: (now: number, was: number) => string;
    vocabLine: (base: number, merges: number, total: number) => string;
    payoff: string;
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
    presets: { liten: string; mellom: string; stor: string };
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
  inspect: {
    title: string;
    intro: string;
    inputLabel: string;
    clickHint: string;
    attnHeading: string;
    attnHelp: string;
    layerLabel: string;
    headLabel: string;
    probHeading: string;
    probHelp: string;
    fasitLabel: string;
    fasitNext: (ch: string) => string;
    correct: string;
    wrong: string;
    noNext: string;
    untrainedHint: string;
    notReady: string;
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
  loss: {
    last: string;
    empty: string;
    axisStep: string;
    axisLoss: string;
    count: (n: number) => string;
  };
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
    para: "Her bygger og trener du en ekte språkmodell – samme type som ChatGPT, bare bitte liten – rett i nettleseren. Du ser den starte med rene tilfeldige tall, gjette, bomme og lære av feilene sine tusenvis av ganger, helt til det begynner å ligne bokmål. Alt skjer lokalt på din maskin, og alt forklares underveis.",
    ctaStart: "Start treningen",
    ctaUnderstand: "Forstå hvordan det fungerer",
    stats: [
      { k: "100 %", v: "lokalt – ingenting forlater maskinen din" },
      { k: "0", v: "forkunnskaper – alt forklares underveis" },
      { k: "ekte", v: "maskinlæring – ikke en animasjon" },
    ],
  },
  understand: {
    title: "Hva er en språkmodell?",
    intro:
      "En språkmodell lærer én enkel ting: å gjette det neste tegnet. Gjør vi det om og om igjen, kan den skrive hele setninger.",
    cards: [
      { t: "Gjett neste tegn", d: "Modellen leser teksten så langt og gjetter hvilken bokstav som bør komme etterpå.", i: "🔮" },
      { t: "Mål feilen", d: "Vi sammenligner gjettingen med den ekte teksten og måler hvor mye den bommet – jo mer overrasket modellen ble, jo større feil. Dette tallet kalles tap (loss).", i: "📏" },
      { t: "Vri på skruene", d: "Inni modellen sitter tusenvis av små justeringsskruer (vektene). Feilen spores bakover, og hver skrue vris bitte litt mot en bedre gjetting – dette kalles backpropagation.", i: "🔧" },
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
    charTooltip: (i) => `tegn #${i}`,
  },
  bpe: {
    title: "Fra tegn til ord-biter",
    intro:
      "Ekte språkmodeller bruker ikke enkeltbokstaver. De lærer «ord-biter» ved å slå sammen de vanligste nabopara igjen og igjen. Prøv selv på den samme teksten. (Den lille modellen vår holder seg til enkelttegn – men nå vet du hvordan de store gjør det.)",
    mergeBtn: "Slå sammen neste par",
    resetBtn: "↺ Nullstill",
    mergeCount: (k, n) => `Sammenslåinger: ${k} / ${n}`,
    thisMergeHeading: "Denne sammenslåingen",
    noMergeYet: "Trykk «Slå sammen neste par» for å starte – akkurat nå er hvert tegn sitt eget token.",
    foundTimes: (n) => `funnet ${n} ganger i teksten`,
    rivalsLabel: "konkurrenter:",
    rulesHeading: "Reglene så langt",
    noRules: "Ingen regler ennå.",
    sentenceHeading: (now, was) => `Setningen nå – ${now} token (var ${was})`,
    vocabLine: (base, merges, total) => `Vokabular: ${base} tegn + ${merges} ord-biter = ${total}`,
    payoff:
      "Når en hel ord-bit blir ett token, ser ikke modellen bokstavene inni – derfor bommer språkmodeller på å telle bokstaver i et ord.",
  },
  arch: {
    title: "Slik er modellen bygd opp",
    intro:
      "Vi bruker en transformer – oppskriften bak moderne språkmodeller som ChatGPT. Teksten renner oppover gjennom blokkene, og hver blokk lærer noe nytt om sammenhengen i teksten.",
    causalTitle: "Ingen juksing:",
    causalBody:
      "når modellen gjetter et tegn, får den bare se det som kom før – aldri fasiten. Slik lærer den å skrive framover. (Fagordet er kausal maskering.)",
    headsTitle: "Flere «blikk» samtidig:",
    headsBody:
      "modellen ser på teksten med flere «hoder» på én gang – ett kan følge med på bokstaver, et annet på ord og betydning. (Kalles multi-head oppmerksomhet.)",
    boxInput: { title: "Inndata", sub: "teksten, delt opp i tegn" },
    boxEmbedding: { title: "Innbygging (embedding)", sub: (dim) => `hvert tegn blir til ${dim} tall` },
    boxBlock: { title: (i) => `Transformer-blokk ${i}`, sub: "ser på sammenhengen + tenker videre" },
    boxAttn: { title: "Ser på sammenhengen", sub: (heads) => `multi-head oppmerksomhet, ${heads} hoder` },
    boxFfn: { title: "Tenker videre", sub: "et lite nevralt nett (feed-forward)" },
    residualNote: "+ snarveier forbi hvert ledd, så ingenting går tapt (residual)",
    boxFinalNorm: { title: "Siste opprydding i tallene", sub: "LayerNorm" },
    boxOutHead: { title: "Poengsum for hvert tegn", sub: "utgangshodet (logits)" },
    boxSoftmax: { title: "Poeng blir sannsynlighet", sub: "softmax" },
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
      "Nå setter vi i gang! For hvert steg gjetter modellen, måler feilen og vrir alle de små skruene bitte litt i riktig retning. Se om tapet synker – da lærer den.",
    modelSize: "Modellstørrelse",
    presets: { liten: "Liten – rask, fin å starte med", mellom: "Mellom – god balanse", stor: "Stor – best resultat, tar lengst tid" },
    minibatch: (n) => `Tekstbiter per steg (minibatch): ${n}`,
    learningRate: (x) => `Skrittlengde (læringsrate): ${x}`,
    start: "▶ Start trening",
    stop: "⏸ Stopp",
    reset: "↺ Nullstill",
    step: (s, max) => `Steg ${s} / ${max}`,
    params: "parametere",
    lossHeading: "Tap (loss) over tid",
    lossHelp:
      "Tapet måler hvor overrasket modellen blir av neste tegn – lavere er bedre. Jo raskere kurven synker, jo fortere lærer modellen. Flater den ut nær null, kan den teksten nesten utenat.",
    liveLabel: "Dette skriver modellen nå",
    livePlaceholder: "Trykk «Start trening» for å se eksempler underveis…",
  },
  chat: {
    title: "Prøv modellen",
    intro:
      "Skriv en starttekst, og la modellen fortsette – ett tegn om gangen. Husk at dette er en bitte liten modell: forvent sjarmerende tull, ikke ChatGPT. Men jo mer du trener den, jo bedre blir det! (Har du ikke trent ennå, blir det bare tilfeldige tegn.)",
    promptLabel: "Din starttekst (bokmål)",
    promptPlaceholder: "f.eks. «Det var en gang»",
    temp: (x) => `Temperatur: ${x}`,
    tempHelp: "0 = trygg, høy = kreativ",
    topK: (k) => `Top-k: ${k}`,
    topKHelp: "hvor mange tegn den får velge blant",
    length: (n) => `Lengde: ${n} tegn`,
    lengthHelp: "hvor mange nye tegn",
    generate: "✨ Generer tekst",
    thinking: "Tenker…",
    answerLabel: "Svar fra modellen",
  },
  inspect: {
    title: "Se inni modellen",
    intro:
      "Nå har modellen lært litt. La oss se hva som skjer inni den for ett enkelt tegn: hva ser den på, og hvilket tegn tror den kommer etterpå?",
    inputLabel: "Tekst å granske",
    clickHint: "Klikk på et tegn under for å velge hvor i teksten du vil se nærmere.",
    attnHeading: "Hva ser modellen på?",
    attnHelp:
      "Hver rad er ett tegn som «ser» bakover. Mørkere rute = mer oppmerksomhet. Det grå feltet er framtiden – den får modellen ikke se.",
    layerLabel: "Lag",
    headLabel: "Hode",
    probHeading: "Hvilket tegn tror modellen kommer nå?",
    probHelp: "Lengre søyle = mer sikker. Dette er det modellen faktisk gjetter på.",
    fasitLabel: "Fasit:",
    fasitNext: (ch) => `det virkelige neste tegnet er «${ch}».`,
    correct: "✓ modellen gjettet riktig!",
    wrong: "✗ modellen bommet denne gangen.",
    noNext: "Dette er det siste tegnet – det finnes ingen fasit å sammenligne med.",
    untrainedHint:
      "Modellen er ikke trent ennå, så den gjetter nesten tilfeldig. Tren den i steget over og kom tilbake hit for å se forskjellen!",
    notReady: "Modellen er ikke klar ennå …",
  },
  rlhf: {
    sectionTitle: "Lær modellen hva du liker (RLHF)",
    sectionIntro:
      "Etter grunntreningen kan du lære modellen smaken din. Den skriver to forslag, du velger det beste – og modellen dyttes litt mot valget ditt. Slik lærte også ChatGPT folkeskikk.",
    introCard:
      "Dette kalles RLHF («Reinforcement Learning from Human Feedback»): mennesker gir tilbakemelding, og modellen justeres mot svarene vi foretrekker. Oppskriften vi bruker heter DPO, og den holder modellen forankret til en frossen kopi av seg selv – så den lærer hva du liker uten å glemme det den allerede kan.",
    startBtn: "Start preferansetrening",
    untrainedHint: "Tips: tren modellen først i steg 4 – da blir fortsettelsene mer meningsfulle.",
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
    dpoLossHeading: "Tap under preferansetreningen (DPO)",
    dpoHelp:
      "Margin = hvor mye modellen har flyttet seg mot valgene dine. Vinner-rate = hvor ofte den nå foretrekker det samme som deg. Høyere er bedre for begge.",
  },
  warning: {
    lead: "Advarsel – ærlig om hva dette er:",
    body:
      " Dette er en svært liten modell som blir trent i nettleseren din på noen få setninger. Den kan ikke måle seg med store modeller som ChatGPT, som er millioner av ganger større og trener i uker på enorme mengder data. Men prinsippet er nøyaktig det samme: ekte transformer, ekte backpropagation, ekte læring. Mer tekst og flere steg gir bedre resultat – prøv å lime inn egen tekst i feltet under!",
  },
  extra: {
    title: "Legg til egen tekst",
    intro:
      "Mer og variert tekst gjør modellen bedre. Lim inn bokmålstekst her (f.eks. fra en bok eller noe du har skrevet). Modellen blir bygd på nytt med de nye dataene.",
    placeholder: "Lim inn bokmålstekst her… (gjerne flere avsnitt)",
    charsNote: (n) => `Tatt med i tillegg til ${n} faste tegn.`,
    rebuild: "Bygg modell på nytt",
  },
  loss: {
    last: "Siste tap:",
    empty: "Start treningen for å se tapet gå ned her.",
    axisStep: "steg →",
    axisLoss: "tap",
    count: (n) => `${n} måling(er)`,
  },
  footer: {
    line1:
      "Bygd med egenskrevet maskinlæringsmotor – transformer, autograd og Adam – helt i JavaScript.",
    line2: "All kode og all læring skjer lokalt i din egen nettleser. 🇳🇴",
  },
  docTitle: "Språkmodell-trener – bygg AI på bokmål",
};

const nn: Strings = {
  header: {
    title: "Språkmodell-trenar",
    subtitle: "Lær AI på nynorsk – i nettlesaren",
    jump: "Hopp til trening →",
  },
  hero: {
    badge: "Ekte trening frå null – ingen ferdig modell",
    h1Pre: "Bygg din eigen språkmodell på",
    h1Lang: "nynorsk",
    para: "Her byggjer og trenar du ein ekte språkmodell – same type som ChatGPT, berre bitte liten – rett i nettlesaren. Du ser han starte med reine tilfeldige tal, gjette, bomme og lære av feila sine tusenvis av gonger, heilt til det byrjar å likne nynorsk. Alt skjer lokalt på maskina di, og alt blir forklart undervegs.",
    ctaStart: "Start treninga",
    ctaUnderstand: "Forstå korleis det fungerer",
    stats: [
      { k: "100 %", v: "lokalt – ingenting forlèt maskina di" },
      { k: "0", v: "forkunnskapar – alt blir forklart undervegs" },
      { k: "ekte", v: "maskinlæring – ikkje ein animasjon" },
    ],
  },
  understand: {
    title: "Kva er ein språkmodell?",
    intro:
      "Ein språkmodell lærer éin enkel ting: å gjette det neste teiknet. Gjer vi det om og om igjen, kan han skrive heile setningar.",
    cards: [
      { t: "Gjett neste teikn", d: "Modellen les teksten så langt og gjettar kva bokstav som bør kome etterpå.", i: "🔮" },
      { t: "Mål feilen", d: "Vi samanliknar gjettinga med den ekte teksten og måler kor mykje han bomma – di meir overraska modellen vart, di større feil. Dette talet blir kalla tap (loss).", i: "📏" },
      { t: "Vri på skruane", d: "Inni modellen sit tusenvis av små justeringsskruar (vektane). Feilen blir spora bakover, og kvar skrue blir vridd bitte litt mot ei betre gjetting – dette blir kalla backpropagation.", i: "🔧" },
    ],
  },
  data: {
    title: "Råtekst og tokenisering",
    intro:
      "Først treng vi tekst. Her bruker vi norsk nynorsk. Datamaskina forstår ikkje bokstavar, så vi deler teksten opp i små einingar – token – og gir kvar av dei eit tal.",
    snippetHeading: "Utsnitt av treningsdataa (nynorsk)",
    charsTotal: (n) => `${n} teikn totalt`,
    howHeading: "Slik blir teksten til tal",
    howPara: (sample) => `Vi deler opp setninga «${sample}» teikn for teikn. Kvart teikn får sin eigen ID:`,
    vocabHeading: (n) => `Heile teiknsettet (${n} token = vokabularet)`,
    charTooltip: (i) => `teikn #${i}`,
  },
  bpe: {
    title: "Frå teikn til ord-bitar",
    intro:
      "Ekte språkmodellar bruker ikkje enkeltbokstavar. Dei lærer «ord-bitar» ved å slå saman dei vanlegaste nabopara om att og om att. Prøv sjølv på den same teksten. (Den vesle modellen vår held seg til enkeltteikn – men no veit du korleis dei store gjer det.)",
    mergeBtn: "Slå saman neste par",
    resetBtn: "↺ Nullstill",
    mergeCount: (k, n) => `Samanslåingar: ${k} / ${n}`,
    thisMergeHeading: "Denne samanslåinga",
    noMergeYet: "Trykk «Slå saman neste par» for å starte – akkurat no er kvart teikn sitt eige token.",
    foundTimes: (n) => `funne ${n} gonger i teksten`,
    rivalsLabel: "konkurrentar:",
    rulesHeading: "Reglane så langt",
    noRules: "Ingen reglar enno.",
    sentenceHeading: (now, was) => `Setninga no – ${now} token (var ${was})`,
    vocabLine: (base, merges, total) => `Vokabular: ${base} teikn + ${merges} ord-bitar = ${total}`,
    payoff:
      "Når ein heil ord-bit blir eitt token, ser ikkje modellen bokstavane inni – difor bommar språkmodellar på å telje bokstavar i eit ord.",
  },
  arch: {
    title: "Slik er modellen bygd opp",
    intro:
      "Vi nyttar ein transformer – oppskrifta bak moderne språkmodellar som ChatGPT. Teksten renn oppover gjennom blokkene, og kvar blokk lærer noko nytt om samanhengen i teksten.",
    causalTitle: "Inga juksing:",
    causalBody:
      "når modellen gjettar eit teikn, får han berre sjå det som kom før – aldri fasiten. Slik lærer han å skrive framover. (Fagordet er kausal maskering.)",
    headsTitle: "Fleire «blikk» samtidig:",
    headsBody:
      "modellen ser på teksten med fleire «hovud» på éin gong – eitt kan følgje med på bokstavar, eit anna på ord og tyding. (Blir kalla multi-head merksemd.)",
    boxInput: { title: "Inndata", sub: "teksten, delt opp i teikn" },
    boxEmbedding: { title: "Innbygging (embedding)", sub: (dim) => `kvart teikn blir til ${dim} tal` },
    boxBlock: { title: (i) => `Transformer-blokk ${i}`, sub: "ser på samanhengen + tenkjer vidare" },
    boxAttn: { title: "Ser på samanhengen", sub: (heads) => `multi-head merksemd, ${heads} hovud` },
    boxFfn: { title: "Tenkjer vidare", sub: "eit lite nevralt nett (feed-forward)" },
    residualNote: "+ snarvegar forbi kvart ledd, så ingenting går tapt (residual)",
    boxFinalNorm: { title: "Siste opprydding i tala", sub: "LayerNorm" },
    boxOutHead: { title: "Poengsum for kvart teikn", sub: "utgangshovudet (logits)" },
    boxSoftmax: { title: "Poeng blir sannsyn", sub: "softmax" },
    explainHeading: "Kva skjer inni?",
    explain: [
      { b: "Innbygging:", t: "kvart teikn blir til ei liste med tal, og vi legg til informasjon om kvar i teksten det står." },
      { b: "Sjølvmerksemd:", t: "kvart teikn ser på dei andre teikna og finn ut kva som er viktig i samanhengen." },
      { b: "Feed-forward:", t: "eit lite nevralt nett som «tenkjer» vidare over kvar posisjon." },
      { b: "Residualvegar:", t: "informasjonen hoppar over kvart ledd slik at ingenting går tapt." },
      { b: "Softmax:", t: "gjer poenga om til sannsyn – slik vel modellen neste teikn." },
    ],
  },
  train: {
    title: "Trening – sjå modellen lære",
    intro:
      "No set vi i gang! For kvart steg gjettar modellen, måler feilen og vrir alle dei små skruane bitte litt i rett retning. Sjå om tapet søkk – då lærer han.",
    modelSize: "Modellstorleik",
    presets: { liten: "Liten – rask, fin å starte med", mellom: "Mellom – god balanse", stor: "Stor – best resultat, tek lengst tid" },
    minibatch: (n) => `Tekstbitar per steg (minibatch): ${n}`,
    learningRate: (x) => `Skrittlengd (læringsrate): ${x}`,
    start: "▶ Start trening",
    stop: "⏸ Stopp",
    reset: "↺ Nullstill",
    step: (s, max) => `Steg ${s} / ${max}`,
    params: "parametrar",
    lossHeading: "Tap (loss) over tid",
    lossHelp:
      "Tapet måler kor overraska modellen blir av neste teikn – lågare er betre. Jo raskare kurva søkk, jo fortare lærer modellen. Flatar ho ut nær null, kan han teksten nesten utanåt.",
    liveLabel: "Dette skriv modellen no",
    livePlaceholder: "Trykk «Start trening» for å sjå døme undervegs…",
  },
  chat: {
    title: "Prøv modellen",
    intro:
      "Skriv ein starttekst, og lat modellen halde fram – eitt teikn om gongen. Hugs at dette er ein bitte liten modell: vent deg sjarmerande tull, ikkje ChatGPT. Men di meir du trenar han, di betre blir det! (Har du ikkje trena enno, blir det berre tilfeldige teikn.)",
    promptLabel: "Din starttekst (nynorsk)",
    promptPlaceholder: "t.d. «Det var ein gong»",
    temp: (x) => `Temperatur: ${x}`,
    tempHelp: "0 = trygg, høg = kreativ",
    topK: (k) => `Top-k: ${k}`,
    topKHelp: "kor mange teikn han får velje mellom",
    length: (n) => `Lengd: ${n} teikn`,
    lengthHelp: "kor mange nye teikn",
    generate: "✨ Generer tekst",
    thinking: "Tenkjer…",
    answerLabel: "Svar frå modellen",
  },
  inspect: {
    title: "Sjå inni modellen",
    intro:
      "No har modellen lært litt. Lat oss sjå kva som skjer inni han for eitt enkelt teikn: kva ser han på, og kva teikn trur han kjem etterpå?",
    inputLabel: "Tekst å granske",
    clickHint: "Klikk på eit teikn under for å velje kvar i teksten du vil sjå nærare.",
    attnHeading: "Kva ser modellen på?",
    attnHelp:
      "Kvar rad er eitt teikn som «ser» bakover. Mørkare rute = meir merksemd. Det grå feltet er framtida – den får modellen ikkje sjå.",
    layerLabel: "Lag",
    headLabel: "Hovud",
    probHeading: "Kva teikn trur modellen kjem no?",
    probHelp: "Lengre søyle = meir sikker. Dette er det modellen faktisk gjettar på.",
    fasitLabel: "Fasit:",
    fasitNext: (ch) => `det verkelege neste teiknet er «${ch}».`,
    correct: "✓ modellen gjetta rett!",
    wrong: "✗ modellen bomma denne gongen.",
    noNext: "Dette er det siste teiknet – det finst ingen fasit å samanlikne med.",
    untrainedHint:
      "Modellen er ikkje trena enno, så han gjettar nesten tilfeldig. Tren han i steget over og kom attende hit for å sjå skilnaden!",
    notReady: "Modellen er ikkje klar enno …",
  },
  rlhf: {
    sectionTitle: "Lær modellen kva du likar (RLHF)",
    sectionIntro:
      "Etter grunntreninga kan du lære modellen smaken din. Han skriv to forslag, du vel det beste – og modellen blir dytta litt mot valet ditt. Slik lærte òg ChatGPT folkeskikk.",
    introCard:
      "Dette blir kalla RLHF («Reinforcement Learning from Human Feedback»): menneske gir tilbakemelding, og modellen blir justert mot svara vi føretrekkjer. Oppskrifta vi bruker heiter DPO, og ho held modellen forankra til ein frosen kopi av seg sjølv – så han lærer kva du likar utan å gløyme det han alt kan.",
    startBtn: "Start preferanse-trening",
    untrainedHint: "Tips: tren modellen først i steg 4 – då blir framhalda meir meiningsfulle.",
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
    dpoLossHeading: "Tap under preferansetreninga (DPO)",
    dpoHelp:
      "Margin = kor mykje modellen har flytta seg mot vala dine. Vinnar-rate = kor ofte han no føretrekkjer det same som deg. Høgare er betre for begge.",
  },
  warning: {
    lead: "Åtvaring – ærleg om kva dette er:",
    body:
      " Dette er ein svært liten modell som blir trent i nettlesaren din på nokre få setningar. Han kan ikkje måle seg med store modellar som ChatGPT, som er millionar av gonger større og trenar i veker på enorme mengder data. Men prinsippet er nøyaktig det same: ekte transformer, ekte backpropagation, ekte læring. Meir tekst og fleire steg gir betre resultat – prøv å lime inn eigen tekst i feltet under!",
  },
  extra: {
    title: "Legg til eigen tekst",
    intro:
      "Meir og variert tekst gjer modellen betre. Lim inn nynorsk tekst her (t.d. frå ei bok eller noko du har skrive). Modellen blir bygd på nytt med dei nye dataa.",
    placeholder: "Lim inn nynorsk tekst her… (gjerne fleire avsnitt)",
    charsNote: (n) => `Teken med i tillegg til ${n} faste teikn.`,
    rebuild: "Bygg modell på nytt",
  },
  loss: {
    last: "Siste tap:",
    empty: "Start treninga for å sjå tapet gå ned her.",
    axisStep: "steg →",
    axisLoss: "tap",
    count: (n) => `${n} måling(ar)`,
  },
  footer: {
    line1:
      "Bygd med eigenskriven maskinlæringsmotor – transformer, autograd og Adam – heilt i JavaScript.",
    line2: "All kode og all læring skjer lokalt i din eigen nettlesar. 🇳🇴",
  },
  docTitle: "Språkmodell-trenar – bygg AI på nynorsk",
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
