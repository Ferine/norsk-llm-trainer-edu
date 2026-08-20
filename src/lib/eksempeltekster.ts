// Eksempeltekster til steg 9 – to hyller i same nedtrekksliste:
//
// KLASSIKARANE (lisens "fri") har falle i det fri (forfattarane døydde for
// meir enn 70 år sidan). Henta ordrett frå Project Gutenberg og Wikisource,
// med tre redigeringar:
//   – harde linjeskift inni avsnitt er løyste opp (avsnitt skil med \n\n)
//   – Gutenbergs « -- » er sett til tankestrek « – », og Vinjes "gaasauge"
//     («...») følgjer norsk skikk
//   – éin openberr OCR-feil i Markens grøde er retta («Fiatbrød» → «Flatbrød»)
// Gamal rettskriving («aa», store substantiv) står som forfattarane skreiv.
//
// MODERNE NORSK (lisens "ccbysa") er henta ordrett frå Wikipedia på bokmål
// («Stor språkmodell», innleiing + Arkitektur) og nynorsk («Maskinlæring»,
// innleiing + Underdomene) 20. august 2026, CC BY-SA 4.0. Berre
// mellomtitlar er fjerna og blankteikn normaliserte; listeprega og
// faktasvake seksjonar er ikkje tekne med. `aar` er hentingsåret –
// artiklane har mange forfattarar og endrar seg over tid.
//
// Alt er baka inn i bygget – appen skal aldri henta noko frå nettet.
// Adressene peikar til kjelda, men blir viste som rein tekst i appen –
// leselista (steg 10) skal framleis vera dei einaste utgåande lenkjene.

export type EksId =
  | "ferdaminni"
  | "sult"
  | "faderen"
  | "jenny"
  | "markens"
  | "wiki-llm"
  | "wiki-ml";

/** Kva for lisensforklaring (i18n `extra.sampleLicense`) som gjeld. */
export type Lisens = "fri" | "ccbysa";

export interface Eksempeltekst {
  id: EksId;
  forfattar: string;
  tittel: string;
  aar: number;
  kjelde: string;
  url: string;
  lisens: Lisens;
  tekst: string;
}

// Kronologisk – frå Vinjes landsmål (1861) til Wikipedia i dag.
export const EKSEMPELTEKSTER: Eksempeltekst[] = [
  {
    id: "ferdaminni",
    forfattar: "Aasmund Olavsson Vinje",
    tittel: "Ferdaminni fraa Sumaren 1860",
    aar: 1861,
    kjelde: "Wikisource",
    url: "https://no.wikisource.org/wiki/Ferdaminni_fraa_Sumaren_1860",
    lisens: "fri",
    tekst: `Mange meina, at aka paa Jarnvegen er leidt og keidt og altid likt seg sjølv; men eg, som er leid af trøyttkjøyrde Øykir og Skranglekjerrur, eg finner det som eit Dikt at fara so fort og sjaa Tre og Steinar og Tuvur og alt, som i Vegen kann koma, at syna seg fram i ein Augneblink og atter renna fraa os som skræmde Fuglar, og so høyra Eimvogni frøsa som ein annan Hest og faa Mat og Drykk af Kol og Vatn. Det er sama Drivkraft som i Hesten og meg og deg berre paa ein liten annan Maate, so det no ikki er værdt at tala um leide og keide Drivverk og slikt nokot, liksom det berre skulde vera Kol og Vatn og Eld og Malm. «Der er ei Livsens Aand i Hjulom», som Profeten talar um.

Det er Mannatanken, som her paa Skaparvis hever blaasit Liv i Nosi paa Jordklumpen, og gjort Kol og Vatn og Eld og Malm til Tenaren sin; derfor skriker og frøser i Fugleflog denne Tenaren under Svipuslaget af vaar Aand, og me sitja paa vaar Sigervogn stoltare enn dei romerske Sigerherrar, og turva ikki som dei hava ein narreklædd Mann bak paa Vogni, som ropar til Manngarden paa Baade Sidur af Sigerskeidet: «Kom i Hug, store Mann, at du maa døy!»

Og so rulla me fram mot Upplysnings og Jordodlings smilande Land, for det var sannt det, som Franskmannen Condorcet alt aatte ti Aar sidan sagde: «Her vil med alt vaart Stræv vera Villmannskap i Livet, til dess me hava lagt under os all Naturkraft, so me faa hena til at gjera for os det tyngste og grøvste Arbeid, so berre me ganga og sjaa etter som Uppsynsmenn.»

Det gjenger dei fleste Folk som meg, at dei slita burt sin beste Ungdomsmerg med at vera Hestar og det endaa tidt paa Ting, som jamvel Hesten er for god til.

Og so sjaa me all den Uppdyrkning af Moar og Myrar og alle dei hyggelege Hus ikring denne Vegen, som eigong vil slengja seg fram berre millom bylgjande Aakrar og blømande Engir.

Nei Jarnvegen er rik paa Tanke og Framhug fraa kver Side, han verdt sedd. Han er ikki lyft upp fraa Jordi, men gjenger som all sann Diktning gjenom det livande Liv, og straar Blomar rund ikring seg.`,
  },
  {
    id: "sult",
    forfattar: "Knut Hamsun",
    tittel: "Sult",
    aar: 1890,
    kjelde: "Project Gutenberg",
    url: "https://www.gutenberg.org/ebooks/30027",
    lisens: "fri",
    tekst: `Det var i den Tid, jeg gik omkring og sulted i Kristiania, denne forunderlige By, som ingen forlader, før han har fået Mærker af den . . . .

Jeg ligger vågen på min Kvist og hører en Klokke nedenunder mig slå seks Slag; det var allerede ganske lyst, og Folk begyndte at færdes op og ned i Trapperne. Nede ved Døren, hvor mit Rum var tapetseret med gamle Numre af »Morgenbladet«, kunde jeg så tydelig se en Bekendtgørelse fra Fyrdirektøren, og lidt tilvenstre derfra et fedt, bugnende Avertissement fra Bager Fabian Olsen om nybagt Brød.

Straks jeg slog Øjnene op, begyndte jeg af gammel Vane at tænke efter, om jeg havde noget at glæde mig til idag. Det havde været lidt knapt for mig i den sidste Tid; den ene efter den anden af mine Ejendele var bragt til »Onkel«, jeg var bleven nervøs og utålsom, et Par Gange havde jeg også ligget tilsengs en Dags Tid af Svimmelhed. Nu og da, når Lykken var god, kunde jeg drive det til at få fem Kroner af et eller andet Blad for en Føljeton.

Det lysned mer og mer, og jeg gav mig til at læse på Avertissementerne nede ved Døren; jeg kunde endog skælne de magre, grinende Bogstaver om »Ligsvøb hos Jomfru Andersen, tilhøjre i Porten«. Det sysselsatte mig en lang Stund, jeg hørte Klokken slå otte nedenunder, inden jeg stod op og klædte mig på.

Jeg åbned Vinduet og så ud. Der, hvor jeg stod, havde jeg Udsigt til en Klædesnor og en åben Mark; langt ude lå Gruen tilbage af en nedbrændt Smedje, hvor nogle Arbejdere var i Færd med at rydde op. Jeg lagde mig med Albuerne ned i Vinduet og stirred ud i Luften. Det blev ganske vist en lys Dag, Høsten var kommet, den fine, svale Årstid, hvori alting skifter Farve og forgår. Støjen var allerede begyndt at lyde i Gaderne og lokked mig ud; dette tomme Værelse, hvis Gulv gynged op og ned for hvert Skridt jeg tog henover det, var som en gisten, uhyggelig Ligkiste; der var ingen ordentlig Lås for Døren og ingen Ovn i Rummet; jeg plejed at ligge på mine Strømper om Natten, forat få dem lidt tørre til om Morgenen. Det eneste, jeg havde at fornøje mig ved, var en liden rød Gyngestol, som jeg sad i om Aftenerne og døsed og tænkte på mangehånde Ting. Når det blæste hårdt, og Dørene nedenunder stod åbne, lød der alleslags underlige Hvin op gennem Gulvet og ind fra Væggene, og »Morgenbladet« nede ved Døren fik Revner så lange som en Hånd.`,
  },
  {
    id: "faderen",
    forfattar: "Arne Garborg",
    tittel: "Den burtkomne Faderen",
    aar: 1899,
    kjelde: "Wikisource",
    url: "https://no.wikisource.org/wiki/Den_burtkomne_Faderen",
    lisens: "fri",
    tekst: `Eg hadde levt som den burtkomne Sonen og var som han komen i Naud; men daa eg som han søkte heim att, var Faderen burte.

Eg gjekk til Prestarne og sagde: «kjære, finne meg att Faderen! for eg hev sett til mi Kraft og mi Von og alt mitt Liv og maa døy; og han er den einaste som kann hjelpa meg.»

Og dei song sine Songar og mana med Bøner og sterke Ord. Men Faderen saag eg ikkje.

Eg gjekk til dei vise og sagde: «kjære, finne meg att Faderen! for eg hev spillt mitt Liv og er i Naud, og er aaleine og rædd, og sjuk; og han er den einaste som kann hjelpa meg.»

Og dei røynde sin Visdom og sine Sjaa-glas, og rekna seg gjenom Himlar og Heimar og ransaka Stjernetaakur og Avgrunnar; men dei sagde: «Faderen er burte, og me finn han ikkje.»

Og ein av dei eldste sagde til meg: «Er du og ein av dei, som leitar etter Faderen? Eg skal segja deg ein Ting. Den som leitar finn, men ikkje det han leitar etter.

Faderen er den, som Baani skapte i sitt Bilæte; og dei gjorde han stor, og sette han upp for seg til ei Trøyst og ei Hjelp; for dei var smaae og kunde ikkje vera faderlause.

Men naar dei vert større, og er ufornøgde med Heimsstelle, og ikkje fær Hjelp naar dei bed, og ikkje Svar naar dei spør, daa gjeng dei paa Leit etter Faderen og finn seg sjølve.»

Eg gjekk til dei som ser Syner og hev Draumar og manar Aander og granskar dulde Ting; og eg sagde: «Kann De visa mig Faderen?»

Men dei viste meg Skuggar. Og Skuggarne svara paa det som eg ikkje spurde um.

Daa vart eg leid og leitte ikkje lenger.

Men daa eg var aaleine i det framande Lande, og tok til aa illtrivast, og Hugen vart uroleg og Dauden banka paa, tenkte eg: um Faderen er burte, vil eg sjaa att Barneheimen. Eg vil finna mine eigne; og dei vil høyra paa meg og tilgjeva meg. Og hjaa dei vil eg finna ei Livd; og der, hjaa mine eigne, vil eg døy, og samlast til mine Fedrar.

Eg gjorde i Pengar det eg enno aatte og drog heim. Men daa eg kom fram var Heimen seld og Ætti spreidd; og Bror min sat som ein Farre paa ei Heid, og dei sagde han var galen. For han hadde selt alt det han aatte og gjeve det til dei fatige.

Daa saag eg at eg var aaleine; og eg hadde ein vond Dag.

Eg gjekk til Bror min og sagde han mi Meining; so leigde eg meg inn hjaa ein framand Mann, til aa bu der. For no vilde eg ikkje flakka lenger.

Og eg sagde med meg sjølv: her vil eg døy.

Eg hev vandra gjenom Live. Eg hev vorte gamall men ikkje vis; eg var rik men hev vorte arm, og no er eg heimkomen men ikke heime. Men endaa vinn eg heim. Det er berre den siste Kneiken att; og dette tronge Rome skal vera min Inngang til Kvila.`,
  },
  {
    id: "jenny",
    forfattar: "Sigrid Undset",
    tittel: "Jenny",
    aar: 1911,
    kjelde: "Project Gutenberg",
    url: "https://www.gutenberg.org/ebooks/32245",
    lisens: "fri",
    tekst: `Musikken kom opover Via Condotti, netop som Helge Gram i skumringen bøiet ind i gaten. Den spillet «Den glade Enke» i et sindssvakt, rivende tempo, saa det klang som vilde fanfarer. Og de svarte smaa soldater stormet forbi i den kolde eftermiddag, mindst som det var en romersk kohorte, der i rasende springmarsch skulde til at styrte sig over barbarernes hærskarer, istedetfor at de ganske fredelig skulde hjem tilkvelds i kasernen. Eller kanske det netop var derfor, de hadde slik fart i sig – tænkte Helge og smilte – for der han stod med frakkekraven brettet op for kulden, hadde han følt en underlig historisk stemning stryke gjennem sig. Men saa tok han til at nynne med: «Nei paa kvinden man aldrig blir klog» – og fortsatte nedover gaten i den retning, som han visste, Corsoen skulde ligge.

Han stanset paa hjørnet og saa opover. – Saa den saa slik ut, Corsoen. En ustanselig rindende strøm av vogner i den trange gaten og et kokende mylder av mennesker paa det smale fortaug.

Han stod stille og saa strømmen rinde forbi sig. Og han smilte, for han tænkte paa, at opover denne gaten kunde han nu drive hver evige kveld i mørkningen gjennem menneskemylderen, til den blev like saa hverdags for ham som Carl Johan hjemme.

Aa han hadde lyst til at gaa og gaa nu med det samme – gjennem alle Roms gater – gjerne hele natten. For han tænkte paa byen, slik den hadde ligget under ham for litt siden, da han stod paa Pincio og saa solen gaa ned.

– Skyer utover hele vesthimmelen, tæt i tæt som smaa lysegraa lam. Og de fik glødende ravgyldne kanter av solen, som sank bakom. Under den bleke himmel laa byen, og Helge visste med et, at akkurat slik maatte Rom se ut – ikke saan, som han hadde drømt sig den, men akkurat slik – slik den var.

Men alt andet han hadde set paa reisen hadde skuffet ham, fordi det ikke var saan, som han hadde tænkt det ut paa forhaand, mens han gik hjemme og længtet etter at komme ut og se det. – Endelig, nu endelig var et syn rikere end alle hans drømme. – Og det var Rom.`,
  },
  {
    id: "markens",
    forfattar: "Knut Hamsun",
    tittel: "Markens grøde",
    aar: 1917,
    kjelde: "Project Gutenberg",
    url: "https://www.gutenberg.org/ebooks/43724",
    lisens: "fri",
    tekst: `Den lange, lange Sti over Myrene og ind i Skogene, hvem har trakket op den? Manden, Mennesket, den første som var her. Det var ingen Sti før ham. Siden fulgte et og andet Dyr de svake Spor over Moer og Myrer og gjorde dem tydeligere, og siden igjen begyndte en og anden Lap at snuse Stien op og gaa den naar han skulde fra Fjæld til Fjæld og se til sin Ren. Slik blev Stien til gjennem den store Almenning som ingen eiet, det herreløse Land.

Manden kommer gaaende mot Nord. Han bærer en Sæk, den første Sæk, den indeholder Niste og nogen Redskaper. Manden er stærk og grov, han har rødt Jærnskjæg og smaa Ar i Ansigtet og paa Hænderne – disse Saartomter, har han faat dem i Arbeide eller i Strid? Han er kanske kommet fra Straf og vil skjule sig, han er kanske Filosof og søker Fred, men ialfald saa kommer han der, et Menneske midt i denne uhyre Ensomhet. Han gaar og gaar, det er stilt for Fugler og Dyr omkring ham, stundom taler han et eller andet Ord med sig selv: Aaja Herregud! sier han. Naar han kommer over Myrene og til venlige Steder med en aapen Slette i Skogen sætter han Sækken ned og begynder at vandre omkring og undersøke Forholdene, efter en Stund kommer han tilbake, tar Sækken paa Ryggen og gaar igjen. Det varer hele Dagen, han ser paa Solen hvad det lider, det blir Nat og han kaster sig paa sin Arm i Lyngen.

Om nogen Timer gaar han igjen, aaja Herregud! gaar igjen ret mot Nord, ser paa Solen hvad det lider, holder Maaltid paa en Leiv Flatbrød og Gjeitost, drikker Vand i en Bæk og fortsætter sin Gang. Ogsaa denne Dag gaar med til hans Vandring, for han maa undersøke saa mange venlige Steder i Skogen. Hvad gaar han efter? Efter Land, efter Jord? Han er kanske en Utvandrer fra Bygderne, han har Øinene med sig og speider, stundom stiger han op paa en Haug og speider. Nu synker Solen igjen.

Han gaar paa Vestsiden af et Dalføre med blandet Skog, her er ogsaa Løvskog og Græsbund, det rækker i Timer, det skumrer, men han hører et lite Sus av en Elv, og dette lille Sus opliver ham som noget levende. Da han kommer op paa Høiden ser han Dalen i Halvmørke nedover og længst ute Himlen mot Sør. Han lægger sig.

Om Morgningen staar han foran et Landskap av Skog og Beitesmark, han stiger ned, her er en grøn Li, han ser et Skimt av Elven langt nede og en Hare som sætter over den i et Sprang. Manden nikker som om det just høver at Elven ikke er bredere end et Sprang. En rugende Rype slaar pludselig op ved hans Føtter og hvæser vildt imot ham, og Manden nikker igjen at her paa Stedet er Dyr og Fugler, det høver atter! Han vandrer i Blaabærlyng og Tyttebærlyng, i den syvtakkede Skogstjærne og i Smaabregner; naar han stanser hist og her og graver med et Jærn i Jorden finder han her Muldjord og der Myr, gjødslet av flere Tusen Aars Løvfald og rotten Kvist. Manden nikker at her slaar han sig ned, jo det gjør han, slaar sig ned. I to Dager vedblir han at streife om i Omegnen, men vender om Kvældene tilbake til Lien. Han sover om Nætterne paa et Barleie, han er blit saa hjemme her, han har alt et Barleie under en Berghammer.`,
  },
  {
    id: "wiki-llm",
    forfattar: "Wikipedia på bokmål",
    tittel: "Stor språkmodell",
    aar: 2026,
    kjelde: "no.wikipedia.org",
    url: "https://no.wikipedia.org/wiki/Stor_spr%C3%A5kmodell",
    lisens: "ccbysa",
    tekst: `En stor språkmodell (engelsk: large language model, LLM) er en avansert maskinlæringsmodell som er spesialisert på å forstå og generere naturlig språk.

En klassisk språkmodell bruker statistikk for å gjette hva et ord er basert på tidligere ord. I en stor språkmodell brukes et nevralt nettverk som er trent opp på forhånd på kjente data, kalt generative pre-trained transformer (GPT).

En stor språkmodell bruker veldig mye tekst for å trene opp det nevrale nettverket, og «stor» peker på antallet parametere og omfanget av data modellen er trent på, gjerne milliarder av ord. Store nevrale nettverk er for kompliserte til at noen kan forutsi hva som skjer, men store språkmodeller sammen med generativ kunstig intelligens (KI), kan likevel gi gode svar.

Store språkmodeller er nesten utelukkende basert på transformer-arkitekturen. Den sentrale mekanismen i denne er self-attention, en metode der hver token i en sekvens beregner en vektet sum av alle andre tokens i samme sekvens. Vektingen styres av innlærte projeksjoner kalt queries, keys og values, som avgjør hvor mye informasjon hver token skal hente fra de øvrige. Denne mekanismen gjør det mulig å fange opp avhengigheter mellom ord uavhengig av avstanden mellom dem i teksten, noe rekurrente nettverk har problemer med når sekvensene blir lange.

Ettersom self-attention behandler alle tokens samtidig, har arkitekturen ingen innebygd informasjon om ordrekkefølgen. For å ivareta rekkefølgen får derfor hver token lagt til en posisjonskoding (engelsk: positional encoding) før den sendes inn i nettverket. Transformeren bruker også flere parallelle self-attention-funksjoner, kalt multi-head attention (flerhodet oppmerksomhet), der hvert «hode» kan spesialisere seg på ulike typer sammenhenger mellom tokens.

Den opprinnelige arkitekturen bestod av en enkoder som leser hele inngangssekvensen samtidig, og en dekoder som genererer utdata én token om gangen. Senere modeller bruker ofte bare én av disse delene: BERT (Bidirectional Encoder Representations from Transformers) bruker kun enkoderen, mens GPT-modellene kun bruker dekoderen.

En praktisk fordel med transformerarkitekturen er at self-attention kan beregnes parallelt for alle posisjoner i en sekvens, i motsetning til rekurrente nettverk der hvert tidssteg avhenger av det forrige. Dette gjør trening på store datamengder vesentlig raskere med moderne maskinvare.`,
  },
  {
    id: "wiki-ml",
    forfattar: "Wikipedia på nynorsk",
    tittel: "Maskinlæring",
    aar: 2026,
    kjelde: "nn.wikipedia.org",
    url: "https://nn.wikipedia.org/wiki/Maskinl%C3%A6ring",
    lisens: "ccbysa",
    tekst: `Maskinlæring (frå engelsk machine learning, ML) er eit fagfelt der ein utviklar, undersøker og nyttar algoritmar som lærer frå data, i staden for å bli programmert med faste reglar. Slike læringsalgoritmar byggjer ofte på statistiske metodar og matematiske metodar, og blir brukte til å løyse problem der det er vanskeleg å spesifisere reglane på førehand, men der ein har eksempeldata som illustrerer ønskte resultat.

Maskinlæring er nært knytt til kunstig intelligens, men ikkje alle maskinlæringsmetodar fell innanfor KI. Faget kombinerer algoritmeteori og teori frå matematisk optimalisering og statistisk modellering.

Ein maskinlæringsmodell er ein databasert representasjon av kunnskap henta frå data – typisk lagra som ei fil med parametrar og struktur som kan lastast inn og brukast for nye førespurnader. Under trening lærer algoritmen ved å konstruere modellen som generaliserer frå døma han får, slik at han kan gjere prediksjonar eller ta avgjerder i nye situasjonar. Etter trening kan modellen brukast til oppgåver som føreseiing, rådgjeving og automatiserte avgjerder – utan at kunnskapen er lagt inn manuelt.

For at ein modell skal reknast som lærande, må han kunne tilpasse seg erfaring og forbetre prestasjonen sin over tid. Det inneber at handlingane modellen legg til grunn, blir vurderte etter resultata dei fører til – slik at positive utfall blir påskjønna og uheldige blir straffa. Sjølve læringsalgoritmen er vanlegvis fastprogrammert, men kan skiftast ut av utviklarar dersom det er behov for vedlikehald, forbetringar eller tilpassing til nye problemstillingar.

Det finst tre hovudtypar av maskinlæring: rettleia læring (supervised learning), ikkje-rettleia læring (unsupervised learning) og forsterkande læring (reinforcement learning).

Ved rettleia læring får algoritmen eit datasett der kvar input er para med ein korrekt output (såkalla «merke» eller label). Målet er at modellen skal lære å forutsjå output for nye, ukjende input basert på tidlegare døme.

I ikkje-rettleia læring er data ikkje merkt – det vil seie at algoritmen får input utan tilhøyrande output. Her prøver modellen å finne skjulte mønster i data, til dømes ved å gruppere liknande datapunkt saman (klynging), eller ved å redusere kompleksiteten i datasettet gjennom teknikkar som dimensjonsreduksjon.

Forsterkande læring skil seg frå dei to andre ved at ein agent lærer gjennom interaksjon med eit miljø. Agenten tek avgjerder og får tilbakemelding i form av belønning eller straff, og målet er å lære ei strategi som maksimerer samla belønning over tid. Denne læringsforma vert mykje brukt innan robotikk og spelutvikling.`,
  },
];
