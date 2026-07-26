// ============================================================================
// Påskeegg: skriv den trente modellen ut som eit rekneark som gjer inferens
// med berre formlar — ingen makroar, ingen VBA.
//
// Tre triks gjer det mogleg:
//
//  1. Alle vektmatriser blir lagra TRANSPONERT. Då blir kvar matrisemultiplikasjon
//     ein SUMPRODUCT mellom to liggjande rader (same form), så vi slepp både
//     MMULT, TRANSPOSE og matriseformlar (Ctrl+Shift+Enter).
//
//  2. Heile genereringa er eitt einaste rutenett med T posisjonar. Posisjon t
//     hentar sin token-id frå argmax av logits på rad t-1. Fordi merksemda er
//     kausalt maskert, ser rad t berre bakover — så det finst ingen sirkulær
//     referanse, og K/V-mellomlagringa («KV-cache») oppstår av seg sjølv: alle
//     radene deler dei same k- og v-cellene.
//
//  3. Den maskerte trekanten blir aldri rekna ut. Cellene finst ikkje. Det gjev
//     halv storleik OG er nødvendig: ein SUMPRODUCT over heile rada ville
//     referert framtidige posisjonar og laga ein sirkel.
//
// Modellen brukar Float32Array; rekneark reknar i float64. Vektene er identiske
// (float32 er eksakt i float64), men mellomrekninga driv litt. Sjå «Les_meg».
// ============================================================================

import {
  newSheet,
  put,
  colName,
  STYLE_BOLD,
  STYLE_WRAP,
  STYLE_MONO,
  type SheetSpec,
  type WorkbookSpec,
} from "./xlsx.js";
import { sampleTokens, type Tensor, type Transformer } from "./ml.js";
import type { Tokenizer } from "./corpus.js";

// ---------------------------------------------------------------- geometri ---

/** Ei utlagd blokk med celler. Gjev A1-referansar til rader, celler og område. */
class Grid {
  constructor(
    readonly sheet: string,
    readonly r0: number,
    readonly c0: number,
    readonly rows: number,
    readonly cols: number
  ) {}

  cell(i: number, j: number): string {
    return `${this.sheet}!$${colName(this.c0 + j)}$${this.r0 + i}`;
  }

  /** Ei liggjande rad, eventuelt avgrensa til kolonnane [j0, j1]. */
  row(i: number, j0 = 0, j1 = this.cols - 1): string {
    const r = this.r0 + i;
    return `${this.sheet}!$${colName(this.c0 + j0)}$${r}:$${colName(this.c0 + j1)}$${r}`;
  }

  /** Heile blokka, absolutt, med arknamn — for definerte namn. */
  area(): string {
    return (
      `${this.sheet}!$${colName(this.c0)}$${this.r0}:` +
      `$${colName(this.c0 + this.cols - 1)}$${this.r0 + this.rows - 1}`
    );
  }

  /** Heile blokka utan arknamn — for vilkårsformatering. */
  local(): string {
    return (
      `${colName(this.c0)}${this.r0}:` +
      `${colName(this.c0 + this.cols - 1)}${this.r0 + this.rows - 1}`
    );
  }
}

interface Cursor {
  row: number;
}

/** Legg ei ny blokk under den førre: tittelrad, så rows×cols med indeks i kol. A. */
function block(
  sh: SheetSpec,
  cur: Cursor,
  title: string,
  rows: number,
  cols: number,
  label = true
): Grid {
  put(sh, cur.row, 1, { s: title, st: STYLE_BOLD });
  cur.row += 1;
  const g = new Grid(sh.name, cur.row, 2, rows, cols);
  if (label) for (let i = 0; i < rows; i++) put(sh, g.r0 + i, 1, { n: i });
  cur.row += rows + 1; // ei blank rad mellom blokkene
  return g;
}

// Lesarar for vektene. T_ gjev den transponerte: T_(W)(i, j) === W[j][i].
const N_ = (m: Tensor) => (i: number, j: number) => m.d[i * m.cols + j];
const T_ = (m: Tensor) => (i: number, j: number) => m.d[j * m.cols + i];

function values(
  sh: SheetSpec,
  cur: Cursor,
  title: string,
  rows: number,
  cols: number,
  get: (i: number, j: number) => number
): Grid {
  const g = block(sh, cur, title, rows, cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) put(sh, g.r0 + i, g.c0 + j, { n: get(i, j) });
  return g;
}

// -------------------------------------------------------------------- tekst ---

export type Lang = "bm" | "nn";

/** Overskriftene inne i rekneblokkene. Same ordval som resten av appen. */
interface Labels {
  weightsIntro: string;
  vocabIntro: (v: number) => string;
  vocabHead: [string, string, string, string];
  embIntro: string;
  embHead: [string, string, string, string];
  layerIntro: (l: number, T: number) => string;
  lnHelp: (which: string) => string;
  ln: (which: string) => string;
  q: string;
  k: string;
  vT: string;
  headScore: (h: number, hd: number) => string;
  headMax: (h: number) => string;
  headExp: (h: number) => string;
  headSum: (h: number) => string;
  headAttn: (h: number) => string;
  headsOut: string;
  attnOut: string;
  res1: string;
  layerOut: (l: number) => string;
  ffnWide: (n: number) => string;
  gelu: string;
  ffnBack: string;
  outIntro: string;
  lastLnHelp: string;
  lastLn: string;
  logits: (v: number) => string;
  maxRow: string;
  winner: string;
  winnerChar: string;
  expRow: string;
  sumRow: string;
  probs: string;
  conf: string;
}

const LABELS: Record<Lang, Labels> = {
  bm: {
    weightsIntro:
      "Alle vekter i modellen. Matrisene er transponert: rad i er kolonne i i den opprinnelige matrisen.",
    vocabIntro: (v) => `Alle ${v} tegn modellen kjenner.`,
    vocabHead: ["id", "tegn", "kodepunkt", "vist som"],
    embIntro:
      "Fra tegn til tall: tegnvektor + posisjonsvektor. Kolonne B henter id-en; posisjoner etter starten henter den fra vinneren på raden over.",
    embHead: ["pos", "id", "tegn", "nytt"],
    layerIntro: (l, T) =>
      `Lag ${l}. Hver blokk har ${T} rader — én per posisjon i teksten. ` +
      `Kolonne A er posisjonen. Klikk en celle for å se regnestykket bak ett tall.`,
    lnHelp: (w) => `${w} hjelp — snitt og 1/spredning per posisjon`,
    ln: (w) => `${w} — normalisert (layernorm)`,
    q: "q = LN1 × Wq (spørsmål)",
    k: "k = LN1 × Wk (nøkkel)",
    vT: "v transponert = (LN1 × Wv)ᵀ (verdi; rad = kanal, kol = posisjon)",
    headScore: (h, hd) => `Hode ${h} — poeng q·k / √${hd} (bare fortiden)`,
    headMax: (h) => `Hode ${h} — største poeng per rad`,
    headExp: (h) => `Hode ${h} — e^(poeng − største)`,
    headSum: (h) => `Hode ${h} — sum av raden`,
    headAttn: (h) =>
      `Hode ${h} — oppmerksomhet (softmax). Rad = hvem som ser, kolonne = hvem som blir sett`,
    headsOut: "Hodene satt sammen: oppmerksomhet × v",
    attnOut: "Oppmerksomhet ut = sammensatt × Wo",
    res1: "Snarvei 1: x + oppmerksomhet (residual)",
    layerOut: (l) => `Snarvei 2 — ut fra lag ${l} (residual)`,
    ffnWide: (n) => `Bredt lag: LN2 × W1 + b1 (${n} kanaler)`,
    gelu: "Myk knekk (GELU) på det brede laget",
    ffnBack: "Tilbake til vanlig bredde: × W2 + b2",
    outIntro:
      "Fra siste lag til et tegn: normaliser, gang med head, ta det største tallet.",
    lastLnHelp: "Siste LN hjelp — snitt og 1/spredning",
    lastLn: "Siste LN (layernorm)",
    logits: (v) => `Poeng per tegn i alfabetet (logits, ${v} kolonner)`,
    maxRow: "Største poeng i raden",
    winner: "Vinneren: id-en til tegnet med flest poeng",
    winnerChar: "Samme id som tegn",
    expRow: "e^(poeng − største)",
    sumRow: "Sum av raden",
    probs: "Sannsynlighet per tegn (softmax) — summerer til 1",
    conf: "Hvor sikker modellen er på vinneren",
  },
  nn: {
    weightsIntro:
      "Alle vekter i modellen. Matrisene er transponerte: rad i er kolonne i i den opphavlege matrisa.",
    vocabIntro: (v) => `Alle ${v} teikn modellen kjenner.`,
    vocabHead: ["id", "teikn", "kodepunkt", "vist som"],
    embIntro:
      "Frå teikn til tal: teiknvektor + posisjonsvektor. Kolonne B hentar id-en; posisjonar etter starten hentar han frå vinnaren på rada over.",
    embHead: ["pos", "id", "teikn", "nytt"],
    layerIntro: (l, T) =>
      `Lag ${l}. Kvar blokk har ${T} rader — éi per posisjon i teksten. ` +
      `Kolonne A er posisjonen. Klikk ei celle for å sjå rekninga bak eitt tal.`,
    lnHelp: (w) => `${w} hjelp — snitt og 1/spreiing per posisjon`,
    ln: (w) => `${w} — normalisert (layernorm)`,
    q: "q = LN1 × Wq (spørsmål)",
    k: "k = LN1 × Wk (nøkkel)",
    vT: "v transponert = (LN1 × Wv)ᵀ (verdi; rad = kanal, kol = posisjon)",
    headScore: (h, hd) => `Hovud ${h} — skår q·k / √${hd} (berre fortida)`,
    headMax: (h) => `Hovud ${h} — største skår per rad`,
    headExp: (h) => `Hovud ${h} — e^(skår − største)`,
    headSum: (h) => `Hovud ${h} — sum av rada`,
    headAttn: (h) =>
      `Hovud ${h} — merksemd (softmax). Rad = kven som ser, kolonne = kven som blir sett`,
    headsOut: "Hovuda sett saman: merksemd × v",
    attnOut: "Merksemd ut = samansett × Wo",
    res1: "Snarveg 1: x + merksemd (residual)",
    layerOut: (l) => `Snarveg 2 — ut frå lag ${l} (residual)`,
    ffnWide: (n) => `Breitt lag: LN2 × W1 + b1 (${n} kanalar)`,
    gelu: "Mjuk knekk (GELU) på det breie laget",
    ffnBack: "Tilbake til vanleg breidd: × W2 + b2",
    outIntro:
      "Frå siste lag til eit teikn: normaliser, gong med head, ta det største talet.",
    lastLnHelp: "Siste LN hjelp — snitt og 1/spreiing",
    lastLn: "Siste LN (layernorm)",
    logits: (v) => `Poeng per teikn i alfabetet (logits, ${v} kolonnar)`,
    maxRow: "Største skår i rada",
    winner: "Vinnaren: id-en til teiknet med flest poeng",
    winnerChar: "Same id som teikn",
    expRow: "e^(skår − største)",
    sumRow: "Sum av rada",
    probs: "Sannsyn per teikn (softmax) — summerer til 1",
    conf: "Kor sikker modellen er på vinnaren",
  },
};

// ------------------------------------------------------------------- vekter ---

interface LayerW {
  ln1g: Grid;
  ln1b: Grid;
  WqT: Grid;
  WkT: Grid;
  WvT: Grid;
  WoT: Grid;
  ln2g: Grid;
  ln2b: Grid;
  W1T: Grid;
  b1: Grid;
  W2T: Grid;
  b2: Grid;
}

interface Weights {
  sheet: SheetSpec;
  tokEmb: Grid;
  posEmb: Grid;
  layers: LayerW[];
  lnFg: Grid;
  lnFb: Grid;
  headT: Grid;
}

function writeWeights(model: Transformer, L: Labels): Weights {
  const sh = newSheet("Vekter");
  sh.cols = [{ min: 1, max: 1, width: 30 }];
  const cur: Cursor = { row: 1 };
  const { dim, vocab, seqLen, ffnMult } = model.cfg;
  const ffn = dim * ffnMult;

  put(sh, cur.row, 1, { s: L.weightsIntro, st: STYLE_BOLD });
  cur.row += 2;

  const tokEmb = values(sh, cur, `tokEmb (${vocab}×${dim})`, vocab, dim, N_(model.tokEmb));
  const posEmb = values(sh, cur, `posEmb (${seqLen}×${dim})`, seqLen, dim, N_(model.posEmb));

  const layers: LayerW[] = model.blocks.map((blk, l) => {
    const p = `L${l}`;
    return {
      ln1g: values(sh, cur, `${p} ln1 gamma (1×${dim})`, 1, dim, N_(blk.ln1g)),
      ln1b: values(sh, cur, `${p} ln1 beta (1×${dim})`, 1, dim, N_(blk.ln1b)),
      WqT: values(sh, cur, `${p} Wq transponert (${dim}×${dim})`, dim, dim, T_(blk.Wq)),
      WkT: values(sh, cur, `${p} Wk transponert (${dim}×${dim})`, dim, dim, T_(blk.Wk)),
      WvT: values(sh, cur, `${p} Wv transponert (${dim}×${dim})`, dim, dim, T_(blk.Wv)),
      WoT: values(sh, cur, `${p} Wo transponert (${dim}×${dim})`, dim, dim, T_(blk.Wo)),
      ln2g: values(sh, cur, `${p} ln2 gamma (1×${dim})`, 1, dim, N_(blk.ln2g)),
      ln2b: values(sh, cur, `${p} ln2 beta (1×${dim})`, 1, dim, N_(blk.ln2b)),
      W1T: values(sh, cur, `${p} W1 transponert (${ffn}×${dim})`, ffn, dim, T_(blk.W1)),
      b1: values(sh, cur, `${p} b1 (1×${ffn})`, 1, ffn, N_(blk.b1)),
      W2T: values(sh, cur, `${p} W2 transponert (${dim}×${ffn})`, dim, ffn, T_(blk.W2)),
      b2: values(sh, cur, `${p} b2 (1×${dim})`, 1, dim, N_(blk.b2)),
    };
  });

  const lnFg = values(sh, cur, `Siste ln gamma (1×${dim})`, 1, dim, N_(model.lnFg));
  const lnFb = values(sh, cur, `Siste ln beta (1×${dim})`, 1, dim, N_(model.lnFb));
  const headT = values(sh, cur, `head transponert (${vocab}×${dim})`, vocab, dim, T_(model.head));

  return { sheet: sh, tokEmb, posEmb, layers, lnFg, lnFb, headT };
}

// -------------------------------------------------------------------- lag ----

interface LayerGrids {
  sheet: SheetSpec;
  out: Grid;
  attn: Grid[]; // merksemd per hovud, for varmekartet
}

function writeLayer(
  model: Transformer,
  l: number,
  xIn: Grid,
  W: LayerW,
  T: number,
  L: Labels
): LayerGrids {
  const { dim, nHead, ffnMult } = model.cfg;
  const hd = dim / nHead;
  const ffn = dim * ffnMult;
  const sh = newSheet(`Lag_${l}`);
  sh.cols = [{ min: 1, max: 1, width: 6 }];
  sh.colorScales = [];
  const cur: Cursor = { row: 1 };

  put(sh, cur.row, 1, { s: L.layerIntro(l, T), st: STYLE_BOLD });
  cur.row += 2;

  // --- normalisering før merksemda ---
  const ln1h = block(sh, cur, L.lnHelp("LN1"), T, 2);
  for (let t = 0; t < T; t++) {
    put(sh, ln1h.r0 + t, ln1h.c0, { f: `AVERAGE(${xIn.row(t)})` });
    put(sh, ln1h.r0 + t, ln1h.c0 + 1, {
      f: `1/SQRT(DEVSQ(${xIn.row(t)})/${dim}+0.00001)`,
    });
  }

  const ln1 = block(sh, cur, L.ln("LN1"), T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, ln1.r0 + t, ln1.c0 + c, {
        f:
          `${W.ln1g.cell(0, c)}*(${xIn.cell(t, c)}-${ln1h.cell(t, 0)})` +
          `*${ln1h.cell(t, 1)}+${W.ln1b.cell(0, c)}`,
      });

  // --- q, k, og v i transponert form ---
  const q = block(sh, cur, L.q, T, dim);
  const k = block(sh, cur, L.k, T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++) {
      put(sh, q.r0 + t, q.c0 + c, { f: `SUMPRODUCT(${ln1.row(t)},${W.WqT.row(c)})` });
      put(sh, k.r0 + t, k.c0 + c, { f: `SUMPRODUCT(${ln1.row(t)},${W.WkT.row(c)})` });
    }

  // v ligg transponert (rad = kanal, kolonne = posisjon) slik at seinare
  // SUMPRODUCT igjen får to liggjande rader.
  const vT = block(sh, cur, L.vT, dim, T);
  for (let c = 0; c < dim; c++)
    for (let t = 0; t < T; t++)
      put(sh, vT.r0 + c, vT.c0 + t, { f: `SUMPRODUCT(${ln1.row(t)},${W.WvT.row(c)})` });

  // --- merksemd per hovud ---
  const attn: Grid[] = [];
  for (let h = 0; h < nHead; h++) {
    const j0 = h * hd;
    const j1 = (h + 1) * hd - 1;

    // Berre den nedre trekanten finst: posisjon t ser aldri framover.
    const sk = block(sh, cur, L.headScore(h, hd), T, T);
    for (let t = 0; t < T; t++)
      for (let u = 0; u <= t; u++)
        put(sh, sk.r0 + t, sk.c0 + u, {
          f: `SUMPRODUCT(${q.row(t, j0, j1)},${k.row(u, j0, j1)})*(1/SQRT(${hd}))`,
        });

    const mx = block(sh, cur, L.headMax(h), T, 1);
    for (let t = 0; t < T; t++)
      put(sh, mx.r0 + t, mx.c0, { f: `MAX(${sk.row(t, 0, t)})` });

    const ex = block(sh, cur, L.headExp(h), T, T);
    for (let t = 0; t < T; t++)
      for (let u = 0; u <= t; u++)
        put(sh, ex.r0 + t, ex.c0 + u, { f: `EXP(${sk.cell(t, u)}-${mx.cell(t, 0)})` });

    const sum = block(sh, cur, L.headSum(h), T, 1);
    for (let t = 0; t < T; t++)
      put(sh, sum.r0 + t, sum.c0, { f: `SUM(${ex.row(t, 0, t)})` });

    const sm = block(sh, cur, L.headAttn(h), T, T);
    for (let t = 0; t < T; t++) {
      for (let u = 0; u <= t; u++)
        put(sh, sm.r0 + t, sm.c0 + u, { f: `${ex.cell(t, u)}/${sum.cell(t, 0)}` });
      for (let u = t + 1; u < T; u++) put(sh, sm.r0 + t, sm.c0 + u, { n: 0 });
    }
    sh.colorScales.push(sm.local());
    attn.push(sm);
  }

  // --- hovuda sett saman, så Wo ---
  const ho = block(sh, cur, L.headsOut, T, dim);
  for (let t = 0; t < T; t++)
    for (let h = 0; h < nHead; h++)
      for (let j = 0; j < hd; j++)
        // Avgrensa til kolonnane 0..t. Utan det ville formelen referert
        // framtidige posisjonar, og Excel ville meldt sirkulær referanse.
        put(sh, ho.r0 + t, ho.c0 + h * hd + j, {
          f: `SUMPRODUCT(${attn[h].row(t, 0, t)},${vT.row(h * hd + j, 0, t)})`,
        });

  const ao = block(sh, cur, L.attnOut, T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, ao.r0 + t, ao.c0 + c, { f: `SUMPRODUCT(${ho.row(t)},${W.WoT.row(c)})` });

  const r1 = block(sh, cur, L.res1, T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, r1.r0 + t, r1.c0 + c, { f: `${xIn.cell(t, c)}+${ao.cell(t, c)}` });

  // --- normalisering før nettverket ---
  const ln2h = block(sh, cur, L.lnHelp("LN2"), T, 2);
  for (let t = 0; t < T; t++) {
    put(sh, ln2h.r0 + t, ln2h.c0, { f: `AVERAGE(${r1.row(t)})` });
    put(sh, ln2h.r0 + t, ln2h.c0 + 1, {
      f: `1/SQRT(DEVSQ(${r1.row(t)})/${dim}+0.00001)`,
    });
  }

  const ln2 = block(sh, cur, L.ln("LN2"), T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, ln2.r0 + t, ln2.c0 + c, {
        f:
          `${W.ln2g.cell(0, c)}*(${r1.cell(t, c)}-${ln2h.cell(t, 0)})` +
          `*${ln2h.cell(t, 1)}+${W.ln2b.cell(0, c)}`,
      });

  // --- det breie nettverket ---
  const h1 = block(sh, cur, L.ffnWide(ffn), T, ffn);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < ffn; j++)
      put(sh, h1.r0 + t, h1.c0 + j, {
        f: `SUMPRODUCT(${ln2.row(t)},${W.W1T.row(j)})+${W.b1.cell(0, j)}`,
      });

  const hg = block(sh, cur, L.gelu, T, ffn);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < ffn; j++) {
      const x = h1.cell(t, j);
      put(sh, hg.r0 + t, hg.c0 + j, {
        f: `0.5*${x}*(1+TANH(SQRT(2/PI())*(${x}+0.044715*${x}*${x}*${x})))`,
      });
    }

  const fo = block(sh, cur, L.ffnBack, T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, fo.r0 + t, fo.c0 + c, {
        f: `SUMPRODUCT(${hg.row(t)},${W.W2T.row(c)})+${W.b2.cell(0, c)}`,
      });

  const r2 = block(sh, cur, L.layerOut(l), T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, r2.r0 + t, r2.c0 + c, { f: `${r1.cell(t, c)}+${fo.cell(t, c)}` });

  return { sheet: sh, out: r2, attn };
}

// ------------------------------------------------------------------ utdata ---

interface OutGrids {
  sheet: SheetSpec;
  logits: Grid;
  next: Grid; // token-id for neste teikn, per posisjon
  char: Grid; // det teiknet, dekoda
}

function writeOutput(
  model: Transformer,
  xIn: Grid,
  W: Weights,
  T: number,
  L: Labels
): OutGrids {
  const { dim, vocab } = model.cfg;
  const sh = newSheet("Utdata");
  sh.cols = [{ min: 1, max: 1, width: 6 }];
  const cur: Cursor = { row: 1 };

  put(sh, cur.row, 1, { s: L.outIntro, st: STYLE_BOLD });
  cur.row += 2;

  const lnh = block(sh, cur, L.lastLnHelp, T, 2);
  for (let t = 0; t < T; t++) {
    put(sh, lnh.r0 + t, lnh.c0, { f: `AVERAGE(${xIn.row(t)})` });
    put(sh, lnh.r0 + t, lnh.c0 + 1, {
      f: `1/SQRT(DEVSQ(${xIn.row(t)})/${dim}+0.00001)`,
    });
  }

  const ln = block(sh, cur, L.lastLn, T, dim);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < dim; c++)
      put(sh, ln.r0 + t, ln.c0 + c, {
        f:
          `${W.lnFg.cell(0, c)}*(${xIn.cell(t, c)}-${lnh.cell(t, 0)})` +
          `*${lnh.cell(t, 1)}+${W.lnFb.cell(0, c)}`,
      });

  const logits = block(sh, cur, L.logits(vocab), T, vocab);
  for (let t = 0; t < T; t++)
    for (let v = 0; v < vocab; v++)
      put(sh, logits.r0 + t, logits.c0 + v, {
        f: `SUMPRODUCT(${ln.row(t)},${W.headT.row(v)})`,
      });

  const mx = block(sh, cur, L.maxRow, T, 1);
  for (let t = 0; t < T; t++)
    put(sh, mx.r0 + t, mx.c0, { f: `MAX(${logits.row(t)})` });

  const next = block(sh, cur, L.winner, T, 1);
  for (let t = 0; t < T; t++)
    put(sh, next.r0 + t, next.c0, {
      f: `MATCH(${mx.cell(t, 0)},${logits.row(t)},0)-1`,
    });

  const char = block(sh, cur, L.winnerChar, T, 1);
  for (let t = 0; t < T; t++)
    put(sh, char.r0 + t, char.c0, { f: `INDEX(Teikn,${next.cell(t, 0)}+1)` });

  // Sannsyn er ikkje nødvendig for å velja teikn, men viser kor sikker modellen er.
  const ex = block(sh, cur, L.expRow, T, vocab);
  for (let t = 0; t < T; t++)
    for (let v = 0; v < vocab; v++)
      put(sh, ex.r0 + t, ex.c0 + v, {
        f: `EXP(${logits.cell(t, v)}-${mx.cell(t, 0)})`,
      });

  const sum = block(sh, cur, L.sumRow, T, 1);
  for (let t = 0; t < T; t++) put(sh, sum.r0 + t, sum.c0, { f: `SUM(${ex.row(t)})` });

  const prob = block(sh, cur, L.probs, T, vocab);
  for (let t = 0; t < T; t++)
    for (let v = 0; v < vocab; v++)
      put(sh, prob.r0 + t, prob.c0 + v, {
        f: `${ex.cell(t, v)}/${sum.cell(t, 0)}`,
      });

  const conf = block(sh, cur, L.conf, T, 1);
  for (let t = 0; t < T; t++)
    put(sh, conf.r0 + t, conf.c0, { f: `MAX(${prob.row(t)})` });

  return { sheet: sh, logits, next, char };
}

// ------------------------------------------------------- framsida (Les_meg) ---

/** Teksten på framsida. Typa så dei to språka ikkje kan koma i utakt. */
interface FrontCopy {
  title: string;
  promptLabel: string;
  outLabel: string;
  refLabel: string;
  sameLabel: string;
  yes: string;
  no: string;
  howTitle: string;
  how1: string;
  how2: string;
  how3: string;
  how4: string;
  how5: string;
  sheetsTitle: string;
  shVokab: string;
  shVekter: string;
  shInn: string;
  shLag: string;
  shUt: string;
  attnTitle: string;
  attnBody: string;
  metaTitle: string;
  mStep: string;
  mLoss: string;
  mParams: string;
  mArch: string;
  mHeads: string;
  mVocab: string;
  mCells: string;
  mPositions: string;
}

const HOW_KEYS = ["how1", "how2", "how3", "how4", "how5"] as const;
const SHEET_KEYS = ["shVokab", "shVekter", "shInn", "shLag", "shUt"] as const;

const COPY: Record<Lang, FrontCopy> = {
  bm: {
    title: "Språkmodellen som regneark",
    promptLabel: "Skriv en start her:",
    outLabel: "Regnearket skriver:",
    refLabel: "Nettleseren skrev:",
    sameLabel: "Likt?",
    yes: "ja",
    no: "nei – se punkt 4 nedenfor",
    howTitle: "Slik bruker du arket",
    how1: "1. Skriv en ny start i cellen ved siden av «Skriv en start her». Bare tegn som finnes i arket «Vokabular» teller.",
    how2: "2. Trykk F9 (Mac: fn+F9) for å regne på nytt. Hele modellen kjører om, og teksten over endrer seg.",
    how3: "3. Det er ingen makroer og ingen VBA her. Alt er vanlige formler. Klikk en celle og se regnestykket.",
    how4: "4. Nettleseren regner med halv presisjon (float32), regnearket med full (float64). Vektene er de samme, men små avrundinger kan hope seg opp og av og til gi et annet tegn. Da er «Likt?» nei — det er ikke en feil, det er presisjon.",
    how5: "5. Arket har plass til et fast antall posisjoner. Blir starten lengre, blir det færre nye tegn.",
    sheetsTitle: "Hva de andre arkene er",
    shVokab: "Vokabular — alle tegnene modellen kan, med nummer.",
    shVekter: "Vekter — alle tallene modellen har lært. Matrisene ligger transponert (se merknad i arket).",
    shInn: "Innebygging — hvert tegn blir en vektor, pluss en vektor for hvor i teksten det står.",
    shLag: "Lag_0, Lag_1 … — selve modellen. Én blokk per steg: normalisering, oppmerksomhet, det brede nettverket.",
    shUt: "Utdata — poeng per tegn, og vinneren.",
    attnTitle: "Tips",
    attnBody: "Oppmerksomhet-blokkene i Lag-arkene er fargelagt. Trekantformen er meningen: en posisjon får aldri se framover.",
    metaTitle: "Om denne filen",
    mStep: "Treningssteg",
    mLoss: "Siste tap",
    mParams: "Antall tall i modellen",
    mArch: "Arkitektur",
    mHeads: "hoder",
    mVocab: "Størrelse på alfabetet",
    mCells: "Formler i arbeidsboken",
    mPositions: "Posisjoner i arket",
  },
  nn: {
    title: "Språkmodellen som rekneark",
    promptLabel: "Skriv ein start her:",
    outLabel: "Reknearket skriv:",
    refLabel: "Nettlesaren skreiv:",
    sameLabel: "Likt?",
    yes: "ja",
    no: "nei – sjå punkt 4 nedanfor",
    howTitle: "Slik brukar du arket",
    how1: "1. Skriv ein ny start i cella ved sida av «Skriv ein start her». Berre teikn som finst i arket «Vokabular» tel.",
    how2: "2. Trykk F9 (Mac: fn+F9) for å rekna på nytt. Heile modellen køyrer om, og teksten over endrar seg.",
    how3: "3. Det finst ingen makroar og ingen VBA her. Alt er vanlege formlar. Klikk ei celle og sjå rekninga.",
    how4: "4. Nettlesaren reknar med halv presisjon (float32), reknearket med full (float64). Vektene er dei same, men små avrundingar kan hopa seg opp og av og til gje eit anna teikn. Då er «Likt?» nei — det er ikkje ein feil, det er presisjon.",
    how5: "5. Arket har plass til eit fast tal posisjonar. Blir starten lengre, blir det færre nye teikn.",
    sheetsTitle: "Kva dei andre arka er",
    shVokab: "Vokabular — alle teikna modellen kan, med nummer.",
    shVekter: "Vekter — alle tala modellen har lært. Matrisene ligg transponerte (sjå merknad i arket).",
    shInn: "Innebygging — kvart teikn blir ein vektor, pluss ein vektor for kvar i teksten det står.",
    shLag: "Lag_0, Lag_1 … — sjølve modellen. Éi blokk per steg: normalisering, merksemd, det breie nettverket.",
    shUt: "Utdata — poeng per teikn, og vinnaren.",
    attnTitle: "Tips",
    attnBody: "Merksemd-blokkene i Lag-arka er fargelagde. Trekantforma er meininga: ein posisjon får aldri sjå framover.",
    metaTitle: "Om denne fila",
    mStep: "Treningssteg",
    mLoss: "Siste tap",
    mParams: "Tal på tal i modellen",
    mArch: "Arkitektur",
    mHeads: "hovud",
    mVocab: "Storleik på alfabetet",
    mCells: "Formlar i arbeidsboka",
    mPositions: "Posisjonar i arket",
  },
};

// ----------------------------------------------------------------- workbook ---

export interface ExcelModelOpts {
  model: Transformer;
  tokenizer: Tokenizer;
  /** Startteksten som blir lagt i den redigerbare cella. */
  prompt: string;
  /** Kor mange teikn arket skal skriva. */
  nGen: number;
  step: number;
  loss: number;
  presetName: string;
  lang: Lang;
}

export interface ExcelModelResult {
  workbook: WorkbookSpec;
  /** Startteksten etter at ukjende teikn er fjerna. */
  prompt: string;
  /** Talet på posisjonar i arket. */
  positions: number;
  /** Teksten nettlesaren produserte for same start (grådig, temperatur 0). */
  reference: string;
  formulaCells: number;
  valueCells: number;
  /** Adresser testane brukar for å rekna ut arket utan Excel. */
  probe: {
    /** Cella som held den ferdige teksten. */
    output: string;
    /** Token-id per posisjon (kolonne B i «Innebygging»). */
    ids: string[];
    /** Vinnar-id per posisjon (kolonne i «Utdata»). */
    next: string[];
    /** Heile logits-rada per posisjon. */
    logitRows: string[];
  };
}

export function buildModelWorkbook(o: ExcelModelOpts): ExcelModelResult {
  const { model, tokenizer } = o;
  const { dim, vocab, nLayer, nHead, seqLen, ffnMult } = model.cfg;
  const C = COPY[o.lang];
  const L = LABELS[o.lang];
  if (tokenizer.vocab !== vocab)
    throw new RangeError(
      `tokenizer (${tokenizer.vocab}) og modell (${vocab}) har ulikt vokabular`
    );

  // Arket slår opp teikn i vokabularet. Fjern det det ikkje kan slå opp, så
  // referansen frå nettlesaren og arket startar på same tekst.
  let prompt = Array.from(o.prompt)
    .filter((ch) => ch.length === 1 && tokenizer.stoi[ch] !== undefined)
    .join("");
  if (prompt.length === 0) prompt = tokenizer.itos[0];
  if (prompt.length > seqLen - 1) prompt = prompt.slice(prompt.length - (seqLen - 1));

  const P = prompt.length;
  const T = Math.min(seqLen, P + Math.max(1, o.nGen) - 1);
  const nOut = T - P + 1;

  // --- vokabular ---
  const vocabSheet = newSheet("Vokabular");
  vocabSheet.cols = [{ min: 1, max: 4, width: 12 }];
  put(vocabSheet, 1, 1, { s: L.vocabIntro(vocab), st: STYLE_BOLD });
  for (const [i, h] of L.vocabHead.entries())
    put(vocabSheet, 2, 1 + i, { s: h, st: STYLE_BOLD });
  for (let i = 0; i < vocab; i++) {
    const ch = tokenizer.itos[i];
    put(vocabSheet, 3 + i, 1, { n: i });
    put(vocabSheet, 3 + i, 2, { s: ch });
    put(vocabSheet, 3 + i, 3, { n: ch.codePointAt(0) ?? 0 });
    put(vocabSheet, 3 + i, 4, { s: ch === " " ? "␣" : ch === "\n" ? "⏎" : ch });
  }
  const teiknRef = `Vokabular!$B$3:$B$${2 + vocab}`;
  const kodeRef = `Vokabular!$C$3:$C$${2 + vocab}`;

  // --- vekter ---
  const W = writeWeights(model, L);

  // --- innebygging (id-formlane blir fylte til slutt) ---
  const emb = newSheet("Innebygging");
  emb.cols = [
    { min: 1, max: 1, width: 6 },
    { min: 2, max: 4, width: 10 },
  ];
  put(emb, 1, 1, { s: L.embIntro, st: STYLE_BOLD });
  for (const [i, h] of L.embHead.entries())
    put(emb, 2, 1 + i, { s: h, st: STYLE_BOLD });
  const x0 = new Grid("Innebygging", 3, 5, T, dim);
  for (let t = 0; t < T; t++) {
    put(emb, 3 + t, 1, { n: t });
    put(emb, 3 + t, 3, { f: `INDEX(Teikn,$B$${3 + t}+1)` });
    for (let c = 0; c < dim; c++)
      put(emb, x0.r0 + t, x0.c0 + c, {
        f: `INDEX(tokEmb,$B$${3 + t}+1,${c + 1})+INDEX(posEmb,$A$${3 + t}+1,${c + 1})`,
      });
  }

  // --- laga ---
  const layerSheets: SheetSpec[] = [];
  let x: Grid = x0;
  for (let l = 0; l < nLayer; l++) {
    const g = writeLayer(model, l, x, W.layers[l], T, L);
    layerSheets.push(g.sheet);
    x = g.out;
  }

  // --- utdata ---
  const out = writeOutput(model, x, W, T, L);

  // --- no kan innebygginga peika på vinnaren frå rada over ---
  for (let t = 0; t < T; t++) {
    const nth = t + 1; // MID er 1-basert
    const lookup = `IFERROR(MATCH(UNICODE(MID(Ledetekst,${nth},1)),Kodepunkt,0)-1,0)`;
    const fallback = t === 0 ? "0" : out.next.cell(t - 1, 0);
    put(emb, 3 + t, 2, { f: `IF(LEN(Ledetekst)>=${nth},${lookup},${fallback})` });
    put(emb, 3 + t, 4, {
      f: `IF(LEN(Ledetekst)>=${nth},"",INDEX(Teikn,$B$${3 + t}+1))`,
    });
  }

  // --- framsida ---
  const front = newSheet("Les_meg");
  front.cols = [
    { min: 1, max: 1, width: 26 },
    { min: 2, max: 2, width: 96 },
  ];
  const reference =
    prompt +
    tokenizer.decode(
      sampleTokens(
        model,
        tokenizer.encode,
        prompt,
        { temperature: 0, topK: 1, length: nOut },
        () => 0
      ).contIds
    );

  // Teksten arket skriv: starten, så «nytt»-kolonna (tom for startposisjonane),
  // så vinnaren på den siste rada.
  const genParts = [
    "Ledetekst",
    ...Array.from({ length: T }, (_, t) => `Innebygging!$D$${3 + t}`),
    out.char.cell(T - 1, 0),
  ];

  let r = 1;
  put(front, r, 1, { s: C.title, st: STYLE_BOLD });
  r += 2;
  put(front, r, 1, { s: C.promptLabel, st: STYLE_BOLD });
  put(front, r, 2, { s: prompt, st: STYLE_MONO });
  const promptRef = `Les_meg!$B$${r}`;
  r += 1;
  put(front, r, 1, { s: C.outLabel, st: STYLE_BOLD });
  put(front, r, 2, { f: genParts.join("&"), st: STYLE_MONO });
  const outRef = `$B$${r}`;
  r += 1;
  put(front, r, 1, { s: C.refLabel });
  put(front, r, 2, { s: reference, st: STYLE_MONO });
  const refRef = `$B$${r}`;
  r += 1;
  put(front, r, 1, { s: C.sameLabel });
  put(front, r, 2, { f: `IF(EXACT(${outRef},${refRef}),"${C.yes}","${C.no}")` });
  r += 2;

  put(front, r++, 1, { s: C.howTitle, st: STYLE_BOLD });
  for (const key of HOW_KEYS)
    put(front, r++, 2, { s: C[key], st: STYLE_WRAP });
  r += 1;

  put(front, r++, 1, { s: C.sheetsTitle, st: STYLE_BOLD });
  for (const key of SHEET_KEYS)
    put(front, r++, 2, { s: C[key], st: STYLE_WRAP });
  r += 1;

  put(front, r, 1, { s: C.attnTitle, st: STYLE_BOLD });
  put(front, r++, 2, { s: C.attnBody, st: STYLE_WRAP });
  r += 1;

  put(front, r++, 1, { s: C.metaTitle, st: STYLE_BOLD });
  const meta: [string, string | number][] = [
    [C.mStep, o.step],
    [C.mLoss, o.loss],
    [C.mParams, model.paramCount()],
    [C.mArch, `${o.presetName}: dim ${dim}, ${nLayer} lag, ${nHead} ${C.mHeads}, ffn ×${ffnMult}`],
    [C.mVocab, vocab],
    [C.mPositions, T],
  ];
  const metaRow0 = r;
  for (const [label, value] of meta) {
    put(front, r, 1, { s: label });
    put(front, r, 2, typeof value === "number" ? { n: value } : { s: value });
    r += 1;
  }

  const sheets = [front, vocabSheet, W.sheet, emb, ...layerSheets, out.sheet];

  // Tel celler, både for «Les_meg» og for testane.
  let formulaCells = 0;
  let valueCells = 0;
  for (const sh of sheets)
    for (const line of sh.cells.values())
      for (const cell of line.values()) {
        if (cell.f !== undefined) formulaCells++;
        else if (cell.n !== undefined) valueCells++;
      }
  put(front, metaRow0 + meta.length, 1, { s: C.mCells });
  put(front, metaRow0 + meta.length, 2, { n: formulaCells });
  valueCells += 1; // cella vi nettopp skreiv

  const workbook: WorkbookSpec = {
    sheets,
    definedNames: [
      { name: "Ledetekst", ref: promptRef },
      { name: "Teikn", ref: teiknRef },
      { name: "Kodepunkt", ref: kodeRef },
      { name: "tokEmb", ref: W.tokEmb.area() },
      { name: "posEmb", ref: W.posEmb.area() },
    ],
  };

  return {
    workbook,
    prompt,
    positions: T,
    reference,
    formulaCells,
    valueCells,
    probe: {
      output: `Les_meg!${outRef}`,
      ids: Array.from({ length: T }, (_, t) => `Innebygging!$B$${3 + t}`),
      next: Array.from({ length: T }, (_, t) => out.next.cell(t, 0)),
      logitRows: Array.from({ length: T }, (_, t) => out.logits.row(t)),
    },
  };
}
