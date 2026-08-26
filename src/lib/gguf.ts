// ============================================================================
// GGUF-eksport: skriv den trente modellen ut i det same filformatet som Llama,
// Mistral og resten av dei store modellane blir distribuerte i.
//
// Fila er ein ekte GGUF v3 – rett magisk tal, nøkkel/verdi-metadata, tensor-
// tabell og justerte F32-data. Ho kan opnast med gguf-dump, gguf-py eller kva
// som helst GGUF-lesar, og då ser du kvar einaste skrue modellen har lært.
//
// Men ho lastar IKKJE i llama.cpp eller ollama, og det er med vilje. Tre ting
// skil oss frå ein modell derifrå, og vi skriv dei ærleg i fila i staden for
// å pynte på dei:
//
//  1. Tokenisatoren vår er på teiknnivå (~70 teikn), ikkje BPE på byte-nivå.
//     llama.cpp ville delt æ, ø og å i to bytes modellen aldri har sett, så
//     norsk tekst ville rakna på fyrste ordet. Difor står det "char" i
//     tokenizer.ggml.model – ukjent for llama.cpp, men sant.
//  2. SiTU-GLU (standardvalet) har ei portgrein gpt2-arkitekturen ikkje har.
//     Då skriv vi arkitekturnamnet "sprakmodell-situ" i staden for å lyge og
//     seie "gpt2". GELU-modellen er derimot ein ekte gpt2 og får heite det.
//  3. Merksemda vår har ingen bias. gpt2 har det. Vi diktar ikkje opp nullar
//     berre for å fylle plassen – tensorane finst rett og slett ikkje.
//
// Tensorane følgjer elles llama.cpp si form og namngjeving nøyaktig, så fila
// er til å kjenne att for eit trent auge.
// ============================================================================

import { expertWidth, ffnWidth, type Expert, type Tensor, type Transformer } from "./ml.js";
import type { Tokenizer } from "./corpus.js";

// ---------------------------- GGUF-primitivar -------------------------------

// "GGUF" lese som eit little-endian u32.
export const GGUF_MAGIC = 0x46554747;
export const GGUF_VERSION = 3;
// Kvar tensor startar på ei 32-byte grense. Det er standardverdien, og vi
// skriv han eksplisitt i general.alignment så lesaren slepp å gjette.
export const GGUF_ALIGNMENT = 32;
// Den einaste tensortypen vi skriv. Ingen kvantisering her – Slankekuren viser
// den historia betre enn ei fil full av 4-bits blokker gjer.
export const GGML_TYPE_F32 = 0;

// Metadata-verditypar, i same rekkjefølgje som spesifikasjonen listar dei.
export const GGUF_TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
} as const;

export type GgufValue =
  | { t: "u32"; v: number }
  | { t: "i32"; v: number }
  | { t: "f32"; v: number }
  | { t: "bool"; v: boolean }
  | { t: "str"; v: string }
  | { t: "strArray"; v: string[] }
  | { t: "i32Array"; v: number[] };

export interface GgufKv {
  key: string;
  val: GgufValue;
}

export interface GgufTensor {
  name: string;
  // ne[] – raskast varierande akse fyrst, same konvensjon som llama.cpp.
  dims: number[];
  data: Float32Array;
}

// Ein liten veksande byte-buffer. All skriving går gjennom DataView med
// eksplisitt little-endian, så fila blir lik uansett kva maskin ho blir laga på.
class ByteWriter {
  private buf: Uint8Array;
  private dv: DataView;
  len = 0;

  constructor(cap = 1 << 16) {
    this.buf = new Uint8Array(cap);
    this.dv = new DataView(this.buf.buffer);
  }

  private need(n: number) {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length;
    while (cap < this.len + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
    this.dv = new DataView(next.buffer);
  }

  u8(v: number) {
    this.need(1);
    this.dv.setUint8(this.len, v);
    this.len += 1;
  }
  u32(v: number) {
    this.need(4);
    this.dv.setUint32(this.len, v, true);
    this.len += 4;
  }
  i32(v: number) {
    this.need(4);
    this.dv.setInt32(this.len, v, true);
    this.len += 4;
  }
  f32(v: number) {
    this.need(4);
    this.dv.setFloat32(this.len, v, true);
    this.len += 4;
  }
  u64(v: number) {
    this.need(8);
    this.dv.setBigUint64(this.len, BigInt(v), true);
    this.len += 8;
  }
  bytes(src: Uint8Array) {
    this.need(src.length);
    this.buf.set(src, this.len);
    this.len += src.length;
  }
  f32Array(src: Float32Array) {
    this.need(src.length * 4);
    for (let i = 0; i < src.length; i++) this.dv.setFloat32(this.len + i * 4, src[i], true);
    this.len += src.length * 4;
  }
  padTo(mult: number) {
    while (this.len % mult !== 0) this.u8(0);
  }
  done(): Uint8Array {
    return this.buf.slice(0, this.len);
  }
}

const enc = new TextEncoder();

// GGUF-streng: lengd som u64, så UTF-8-bytes. Ingen null-terminator.
function writeStr(w: ByteWriter, s: string) {
  const b = enc.encode(s);
  w.u64(b.length);
  w.bytes(b);
}

function writeValue(w: ByteWriter, val: GgufValue) {
  switch (val.t) {
    case "u32":
      w.u32(GGUF_TYPE.UINT32);
      w.u32(val.v);
      break;
    case "i32":
      w.u32(GGUF_TYPE.INT32);
      w.i32(val.v);
      break;
    case "f32":
      w.u32(GGUF_TYPE.FLOAT32);
      w.f32(val.v);
      break;
    case "bool":
      w.u32(GGUF_TYPE.BOOL);
      w.u8(val.v ? 1 : 0);
      break;
    case "str":
      w.u32(GGUF_TYPE.STRING);
      writeStr(w, val.v);
      break;
    case "strArray":
      w.u32(GGUF_TYPE.ARRAY);
      w.u32(GGUF_TYPE.STRING);
      w.u64(val.v.length);
      for (const s of val.v) writeStr(w, s);
      break;
    case "i32Array":
      w.u32(GGUF_TYPE.ARRAY);
      w.u32(GGUF_TYPE.INT32);
      w.u64(val.v.length);
      for (const n of val.v) w.i32(n);
      break;
  }
}

function alignUp(n: number, mult: number): number {
  const rest = n % mult;
  return rest === 0 ? n : n + (mult - rest);
}

// Set saman ei komplett GGUF-fil: hovud, metadata, tensortabell, så data.
export function writeGguf(kv: GgufKv[], tensors: GgufTensor[]): Uint8Array {
  for (const t of tensors) {
    if (t.dims.length < 1) throw new RangeError(`tensor ${t.name} has no dimensions`);
    let n = 1;
    for (const d of t.dims) {
      if (!Number.isInteger(d) || d < 1)
        throw new RangeError(`tensor ${t.name} has a bad dimension ${d}`);
      n *= d;
    }
    if (n !== t.data.length)
      throw new RangeError(
        `tensor ${t.name}: dims say ${n} elements, data has ${t.data.length}`
      );
  }

  const w = new ByteWriter();
  w.u32(GGUF_MAGIC);
  w.u32(GGUF_VERSION);
  w.u64(tensors.length);
  w.u64(kv.length);
  for (const { key, val } of kv) {
    writeStr(w, key);
    writeValue(w, val);
  }

  // Offseta i tabellen er relative til starten av datablokka, ikkje til starten
  // av fila, og kvar tensor må byrje på ei justert grense.
  const offsets: number[] = [];
  let off = 0;
  for (const t of tensors) {
    offsets.push(off);
    off = alignUp(off + t.data.length * 4, GGUF_ALIGNMENT);
  }

  tensors.forEach((t, i) => {
    writeStr(w, t.name);
    w.u32(t.dims.length);
    for (const d of t.dims) w.u64(d);
    w.u32(GGML_TYPE_F32);
    w.u64(offsets[i]);
  });

  // Sjølve datablokka startar justert, og kvar tensor blir lagt nøyaktig der
  // tabellen lova at ho ligg.
  w.padTo(GGUF_ALIGNMENT);
  const base = w.len;
  tensors.forEach((t, i) => {
    while (w.len - base < offsets[i]) w.u8(0);
    w.f32Array(t.data);
  });
  w.padTo(GGUF_ALIGNMENT);

  return w.done();
}

// ------------------------- modell → tensorar --------------------------------

// Rad-major [rows, cols] → rad-major [cols, rows].
function transposed(d: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(d.length);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) out[c * rows + r] = d[r * cols + c];
  return out;
}

// Vekttavle: vår [tal, dim] ligg allereie slik llama.cpp vil ha henne, med dim
// som raskast varierande akse. Ingen transponering.
function embedTensor(name: string, p: Tensor): GgufTensor {
  return { name, dims: [p.cols, p.rows], data: p.d.slice() };
}

// Lineært lag: vi reknar matmul(x, W) med W som [inn, ut]. llama.cpp vil ha
// ne = [inn, ut], altså inn som raskast varierande – som er transponert av vår.
function linearTensor(name: string, p: Tensor): GgufTensor {
  return { name, dims: [p.rows, p.cols], data: transposed(p.d, p.rows, p.cols) };
}

// Bias og normaliseringsvekter: éin dimensjon, ingen omforming.
function vecTensor(name: string, p: Tensor): GgufTensor {
  return { name, dims: [p.cols], data: p.d.slice() };
}

// gpt2 har spørsmål, nøkkel og verdi slått saman til éi matrise. Kvar av dei tre
// blir transponert for seg og lagd etter kvarandre langs ut-aksen.
function qkvTensor(name: string, wq: Tensor, wk: Tensor, wv: Tensor): GgufTensor {
  const dim = wq.rows;
  const out = new Float32Array(3 * dim * dim);
  [wq, wk, wv].forEach((p, i) => {
    out.set(transposed(p.d, p.rows, p.cols), i * dim * dim);
  });
  return { name, dims: [dim, 3 * dim], data: out };
}

export interface GgufBuildOpts {
  model: Transformer;
  tokenizer: Tokenizer;
  step: number;
  loss: number;
  presetName: string;
  lang: string;
}

export interface GgufBuild {
  bytes: Uint8Array;
  arch: string;
  tensorCount: number;
  kvCount: number;
}

// Arkitekturnamnet fila melder om seg sjølv. GELU-modellen ER ein gpt2; SiTU
// er noko anna, og då seier vi det. Ekspertar er eit endå større avvik, og
// vinn over aktiveringa i namnet – aktiveringa står uansett i ein eigen nøkkel.
export function archName(model: Transformer): string {
  if (model.ngram) return model.moe ? "sprakmodell-ngram-moe" : "sprakmodell-ngram";
  if (model.moe) return "sprakmodell-moe";
  return model.act === "situ" ? "sprakmodell-situ" : "gpt2";
}

export function buildModelGguf(o: GgufBuildOpts): GgufBuild {
  const { model, tokenizer } = o;
  const cfg = model.cfg;
  const arch = archName(model);
  const situ = model.act === "situ";

  const tensors: GgufTensor[] = [
    embedTensor("token_embd.weight", model.tokEmb),
    embedTensor("position_embd.weight", model.posEmb),
  ];
  if (model.ngramEmb) tensors.push(embedTensor("ngram_embd.weight", model.ngramEmb));

  model.blocks.forEach((blk, i) => {
    const p = `blk.${i}`;
    tensors.push(
      vecTensor(`${p}.attn_norm.weight`, blk.ln1g),
      vecTensor(`${p}.attn_norm.bias`, blk.ln1b),
      qkvTensor(`${p}.attn_qkv.weight`, blk.Wq, blk.Wk, blk.Wv),
      linearTensor(`${p}.attn_output.weight`, blk.Wo),
      vecTensor(`${p}.ffn_norm.weight`, blk.ln2g),
      vecTensor(`${p}.ffn_norm.bias`, blk.ln2b)
    );
    // Eitt breitt lag, skrive ut med sitt eige namneprefiks. Den delte
    // eksperten er blokka sitt eige lag og held namna frå før; dei ruta får
    // eit nummer, slik at ei fil med ekspertar ikkje kan forvekslast med ei utan.
    const pushFfn = (q: string, e: Expert) => {
      if (situ && e.Wu && e.bu) {
        // W1 er portgreina, Wu er oppgreina – same rollefordeling som llama.cpp
        // gjev ffn_gate og ffn_up i dei gata modellane sine.
        tensors.push(
          linearTensor(`${q}.ffn_gate.weight`, e.W1),
          vecTensor(`${q}.ffn_gate.bias`, e.b1),
          linearTensor(`${q}.ffn_up.weight`, e.Wu),
          vecTensor(`${q}.ffn_up.bias`, e.bu)
        );
      } else {
        tensors.push(
          linearTensor(`${q}.ffn_up.weight`, e.W1),
          vecTensor(`${q}.ffn_up.bias`, e.b1)
        );
      }
      tensors.push(
        linearTensor(`${q}.ffn_down.weight`, e.W2),
        vecTensor(`${q}.ffn_down.bias`, e.b2)
      );
    };
    pushFfn(p, blk);
    if (blk.router) {
      // ffn_gate_inp er llama.cpp sitt namn på nettopp rutar-matrisa.
      tensors.push(linearTensor(`${p}.ffn_gate_inp.weight`, blk.router.W));
      (blk.routed ?? []).forEach((e, n) => pushFfn(`${p}.exp${n}`, e));
    }
  });

  tensors.push(
    vecTensor("output_norm.weight", model.lnFg),
    vecTensor("output_norm.bias", model.lnFb),
    linearTensor("output.weight", model.head)
  );

  const name = `sprakmodell-${o.presetName}-steg${o.step}`;
  const kv: GgufKv[] = [
    { key: "general.architecture", val: { t: "str", v: arch } },
    { key: "general.name", val: { t: "str", v: name } },
    { key: "general.alignment", val: { t: "u32", v: GGUF_ALIGNMENT } },
    // 0 = alle tensorar i F32.
    { key: "general.file_type", val: { t: "u32", v: 0 } },
    {
      key: "general.description",
      val: {
        t: "str",
        v:
          "Trena i nettlesaren av den norske språkmodell-treneren. Ekte GGUF, " +
          "men lastar ikkje i llama.cpp: tokenisatoren er på teiknnivå og " +
          "merksemda har ingen bias." +
          (situ ? " Det breie laget bruker SiTU-GLU, ikkje GELU." : "") +
          (cfg.moe
            ? ` Det breie laget er delt i ${cfg.moe.experts} ruta ekspertar` +
              ` pluss éin delt; ${cfg.moe.topK} av dei ruta reknar per token.`
            : "") +
          (cfg.ngram
            ? ` Eit ${cfg.ngram.size}-gramminne hash-ar vanlege teikn-ID-ar til` +
              ` éi av ${cfg.ngram.slots} rader før blokk ${cfg.ngram.layer + 1};` +
              " det lagar ingen nye token."
            : ""),
      },
    },

    { key: `${arch}.context_length`, val: { t: "u32", v: cfg.seqLen } },
    { key: `${arch}.embedding_length`, val: { t: "u32", v: cfg.dim } },
    { key: `${arch}.block_count`, val: { t: "u32", v: cfg.nLayer } },
    // Med ekspertar er breidda per ekspert det tala i tensorane faktisk er.
    { key: `${arch}.feed_forward_length`, val: { t: "u32", v: expertWidth(cfg) } },
    ...(cfg.moe
      ? ([
          { key: `${arch}.expert_count`, val: { t: "u32", v: cfg.moe.experts } },
          { key: `${arch}.expert_used_count`, val: { t: "u32", v: cfg.moe.topK } },
          { key: `${arch}.expert_shared_count`, val: { t: "u32", v: 1 } },
          { key: `${arch}.expert_feed_forward_length`, val: { t: "u32", v: expertWidth(cfg) } },
          // Totalbreidda før delinga – slik at ein kan sjå at ingen skruer kom til.
          { key: "sprakmodell.ffn_total_width", val: { t: "u32", v: ffnWidth(cfg) } },
        ] as GgufKv[])
      : []),
    ...(cfg.ngram
      ? ([
          { key: `${arch}.ngram.size`, val: { t: "u32", v: cfg.ngram.size } },
          { key: `${arch}.ngram.bucket_count`, val: { t: "u32", v: cfg.ngram.slots } },
          { key: `${arch}.ngram.layer`, val: { t: "u32", v: cfg.ngram.layer } },
          {
            key: `${arch}.ngram.hash`,
            val: { t: "str", v: "fnv1a-u32-token-ids-vocab-as-bos" },
          },
        ] as GgufKv[])
      : []),
    { key: `${arch}.attention.head_count`, val: { t: "u32", v: cfg.nHead } },
    { key: `${arch}.attention.layer_norm_epsilon`, val: { t: "f32", v: 1e-5 } },

    // "char" er ikkje ein tokenisator llama.cpp kjenner. Det er heile poenget:
    // eitt teikn er éin token her, og det skal fila seie.
    { key: "tokenizer.ggml.model", val: { t: "str", v: "char" } },
    { key: "tokenizer.ggml.tokens", val: { t: "strArray", v: tokenizer.itos.slice() } },
    {
      key: "tokenizer.ggml.token_type",
      // 1 = NORMAL. Vi har ingen spesialtokens i det heile.
      val: { t: "i32Array", v: tokenizer.itos.map(() => 1) },
    },

    { key: "sprakmodell.training.steps", val: { t: "u32", v: o.step } },
    { key: "sprakmodell.training.loss", val: { t: "f32", v: o.loss } },
    { key: "sprakmodell.training.preset", val: { t: "str", v: o.presetName } },
    { key: "sprakmodell.training.language", val: { t: "str", v: o.lang } },
    { key: "sprakmodell.training.activation", val: { t: "str", v: model.act } },
    { key: "sprakmodell.parameter_count", val: { t: "u32", v: model.paramCount() } },
  ];

  return { bytes: writeGguf(kv, tensors), arch, tensorCount: tensors.length, kvCount: kv.length };
}

export function ggufToBlob(bytes: Uint8Array): Blob {
  // Kopien held blobben trygg om bufferet skulle bli attbrukt seinare.
  return new Blob([bytes.slice().buffer as ArrayBuffer], { type: "application/octet-stream" });
}

export async function downloadGguf(bytes: Uint8Array, filename: string): Promise<void> {
  const url = URL.createObjectURL(ggufToBlob(bytes));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Gje nettlesaren eit augeblink til å starte nedlastinga før vi slepp URL-en.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
