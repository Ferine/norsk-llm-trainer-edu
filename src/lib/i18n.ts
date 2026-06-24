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
    charTooltip: (i) => `tegn #${i}`,
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
    presets: { liten: "Liten – raskest", mellom: "Mellom – balanse", stor: "Stor – tregest" },
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
  inspect: {
    title: "Se inni modellen",
    intro:
      "Nå har modellen lært litt. La oss se hva som skjer inni den for ett enkelt tegn: hva ser den på, og hva tror den kommer neste?",
    inputLabel: "Tekst å granske",
    clickHint: "Klikk på et tegn under for å velge hvor i teksten du vil se nærmere.",
    attnHeading: "Hva ser modellen på?",
    attnHelp:
      "Hver rad er ett tegn som «ser» bakover. Mørkere rute = mer oppmerksomhet. Det grå feltet er framtiden – den får modellen ikke se.",
    layerLabel: "Lag",
    headLabel: "Hode",
    probHeading: "Hva tror modellen kommer neste?",
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
  loss: {
    last: "Siste tap:",
    empty: "Start treningen for å se tapet gå ned her.",
    axisStep: "steg →",
    axisLoss: "tap",
    count: (n) => `${n} måling(er)`,
  },
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
    charTooltip: (i) => `teikn #${i}`,
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
    presets: { liten: "Liten – raskast", mellom: "Mellom – balanse", stor: "Stor – tregast" },
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
  inspect: {
    title: "Sjå inni modellen",
    intro:
      "No har modellen lært litt. Lat oss sjå kva som skjer inni han for eitt enkelt teikn: kva ser han på, og kva trur han kjem neste?",
    inputLabel: "Tekst å granske",
    clickHint: "Klikk på eit teikn under for å velje kvar i teksten du vil sjå nærare.",
    attnHeading: "Kva ser modellen på?",
    attnHelp:
      "Kvar rad er eitt teikn som «ser» bakover. Mørkare rute = meir merksemd. Det grå feltet er framtida – den får modellen ikkje sjå.",
    layerLabel: "Lag",
    headLabel: "Hovud",
    probHeading: "Kva trur modellen kjem neste?",
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
  loss: {
    last: "Siste tap:",
    empty: "Start treninga for å sjå tapet gå ned her.",
    axisStep: "steg →",
    axisLoss: "tap",
    count: (n) => `${n} måling(ar)`,
  },
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
