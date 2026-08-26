// Exercises the GGUF writer. The file is parsed back by a reader written from
// the spec alone, not from the writer's own code, so a wrong magic, a bad
// alignment, a lying tensor offset or a byte-order slip surfaces here rather
// than in gguf-dump.

import assert from "node:assert/strict";
import { buildModelGguf, writeGguf, ggufToBlob, downloadGguf, GGUF_ALIGNMENT } from "./dist/gguf.js";
import { Transformer, mulberry32, ffnWidth, trainStep, Adam } from "./dist/ml.js";
import { buildTokenizer, corpora } from "./dist/corpus.js";

// ---- a hand-rolled GGUF reader, so the test does not trust the writer ------

const VT = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

function readGguf(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = 0;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const i32 = () => { const v = dv.getInt32(p, true); p += 4; return v; };
  const f32 = () => { const v = dv.getFloat32(p, true); p += 4; return v; };
  const u64 = () => { const v = dv.getBigUint64(p, true); p += 8; return Number(v); };
  const str = () => {
    const n = u64();
    const s = new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(p, p + n));
    p += n;
    return s;
  };

  assert.equal(u32(), 0x46554747, "magic should read as GGUF");
  assert.equal(u32(), 3, "should be GGUF version 3");
  const tensorCount = u64();
  const kvCount = u64();

  const readValue = (type) => {
    switch (type) {
      case VT.UINT32: return u32();
      case VT.INT32: return i32();
      case VT.FLOAT32: return f32();
      case VT.BOOL: { const v = dv.getUint8(p); p += 1; return v !== 0; }
      case VT.STRING: return str();
      case VT.ARRAY: {
        const elem = u32();
        const n = u64();
        const out = [];
        for (let i = 0; i < n; i++) out.push(readValue(elem));
        return out;
      }
      default: throw new Error(`unhandled metadata type ${type}`);
    }
  };

  const kv = new Map();
  for (let i = 0; i < kvCount; i++) {
    const key = str();
    assert.ok(!kv.has(key), `duplicate metadata key ${key}`);
    kv.set(key, readValue(u32()));
  }

  const infos = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = str();
    const nDims = u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(u64());
    const type = u32();
    const offset = u64();
    infos.push({ name, dims, type, offset });
  }

  const align = kv.get("general.alignment") ?? 32;
  const base = p % align === 0 ? p : p + (align - (p % align));

  const tensors = new Map();
  for (const info of infos) {
    assert.equal(info.type, 0, `${info.name} should be F32`);
    assert.equal(info.offset % align, 0, `${info.name} offset must be aligned`);
    const n = info.dims.reduce((a, b) => a * b, 1);
    const at = base + info.offset;
    assert.ok(at + n * 4 <= bytes.length, `${info.name} runs past the end of the file`);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = dv.getFloat32(at + i * 4, true);
    assert.ok(!tensors.has(info.name), `duplicate tensor ${info.name}`);
    tensors.set(info.name, { dims: info.dims, data });
  }

  return { kv, tensors, dataStart: base };
}

// ---- a small trained model, so the bytes are not all zeros ------------------

const tok = buildTokenizer(corpora.bm);
const data = tok.encode(corpora.bm.slice(0, 4000));

function train(act, steps) {
  const cfg = { vocab: tok.vocab, dim: 16, nLayer: 2, nHead: 2, seqLen: 12, ffnMult: 4, act };
  const model = new Transformer(cfg, mulberry32(7));
  const opt = new Adam(model.params, 0.002);
  for (let i = 0; i < steps; i++) trainStep(model, opt, data, cfg.seqLen, 2, mulberry32(100 + i));
  return { cfg, model };
}

// ---- round-trip both activations ------------------------------------------

for (const act of ["gelu", "situ"]) {
  const { cfg, model } = train(act, 3);
  const built = buildModelGguf({
    model,
    tokenizer: tok,
    step: 3,
    loss: 3.25,
    presetName: "liten",
    lang: "bm",
  });
  const back = readGguf(built.bytes);

  assert.equal(back.tensors.size, built.tensorCount, "tensor count should match the header");
  assert.equal(back.kv.size, built.kvCount, "metadata count should match the header");
  assert.equal(back.dataStart % GGUF_ALIGNMENT, 0, "tensor data must start aligned");

  // --- metadata says what the model actually is ---
  const arch = act === "situ" ? "sprakmodell-situ" : "gpt2";
  assert.equal(back.kv.get("general.architecture"), arch);
  assert.equal(back.kv.get("general.name"), "sprakmodell-liten-steg3");
  assert.equal(back.kv.get("general.file_type"), 0);
  assert.equal(back.kv.get(`${arch}.context_length`), cfg.seqLen);
  assert.equal(back.kv.get(`${arch}.embedding_length`), cfg.dim);
  assert.equal(back.kv.get(`${arch}.block_count`), cfg.nLayer);
  assert.equal(back.kv.get(`${arch}.attention.head_count`), cfg.nHead);
  assert.equal(back.kv.get(`${arch}.feed_forward_length`), ffnWidth(cfg));
  assert.ok(
    Math.abs(back.kv.get(`${arch}.attention.layer_norm_epsilon`) - 1e-5) < 1e-12,
    "layer-norm epsilon should survive as f32"
  );
  assert.equal(back.kv.get("sprakmodell.training.activation"), act);
  assert.equal(back.kv.get("sprakmodell.training.steps"), 3);
  assert.equal(back.kv.get("sprakmodell.parameter_count"), model.paramCount());

  // --- the tokenizer is written honestly, and survives æøå ---
  assert.equal(back.kv.get("tokenizer.ggml.model"), "char");
  const toks = back.kv.get("tokenizer.ggml.tokens");
  assert.deepEqual(toks, tok.itos, "every character should come back byte-identical");
  assert.equal(back.kv.get("tokenizer.ggml.token_type").length, tok.vocab);
  for (const ch of ["æ", "ø", "å"])
    if (tok.itos.includes(ch))
      assert.ok(toks.includes(ch), `${ch} should survive the UTF-8 round trip`);

  // --- shapes follow the llama.cpp convention: ne[0] varies fastest ---
  const shape = (n) => back.tensors.get(n).dims;
  assert.deepEqual(shape("token_embd.weight"), [cfg.dim, tok.vocab]);
  assert.deepEqual(shape("position_embd.weight"), [cfg.dim, cfg.seqLen]);
  assert.deepEqual(shape("blk.0.attn_qkv.weight"), [cfg.dim, 3 * cfg.dim]);
  assert.deepEqual(shape("blk.0.attn_output.weight"), [cfg.dim, cfg.dim]);
  assert.deepEqual(shape("blk.0.attn_norm.weight"), [cfg.dim]);
  assert.deepEqual(shape("blk.0.ffn_down.weight"), [ffnWidth(cfg), cfg.dim]);
  assert.deepEqual(shape("output.weight"), [cfg.dim, tok.vocab]);
  assert.deepEqual(shape("output_norm.bias"), [cfg.dim]);

  // The gated branch exists only for SiTU, and never for GELU.
  assert.equal(back.tensors.has("blk.0.ffn_gate.weight"), act === "situ");
  assert.deepEqual(shape("blk.0.ffn_up.weight"), [cfg.dim, ffnWidth(cfg)]);
  // Attention has no bias in this model, so we must not have invented one.
  assert.ok(!back.tensors.has("blk.0.attn_qkv.bias"), "must not fabricate an attention bias");
  assert.ok(!back.tensors.has("blk.0.attn_output.bias"), "must not fabricate an output bias");

  // --- the numbers themselves ---
  // Embeddings keep their layout, so they must come back element-for-element.
  assert.deepEqual(
    Array.from(back.tensors.get("token_embd.weight").data),
    Array.from(model.tokEmb.d),
    "token embeddings should round-trip untouched"
  );

  // Linear weights are transposed on the way out; undo it and compare.
  const blk = model.blocks[0];
  const wo = back.tensors.get("blk.0.attn_output.weight").data;
  for (let r = 0; r < blk.Wo.rows; r++)
    for (let c = 0; c < blk.Wo.cols; c++)
      assert.equal(wo[c * blk.Wo.rows + r], blk.Wo.d[r * blk.Wo.cols + c],
        `attn_output[${r},${c}] should be transposed, not scrambled`);

  // Q, K and V must land in that order inside the fused matrix.
  const qkv = back.tensors.get("blk.0.attn_qkv.weight").data;
  [blk.Wq, blk.Wk, blk.Wv].forEach((p, i) => {
    const off = i * cfg.dim * cfg.dim;
    for (let r = 0; r < p.rows; r++)
      for (let c = 0; c < p.cols; c++)
        assert.equal(qkv[off + c * p.rows + r], p.d[r * p.cols + c],
          `qkv slice ${i} at [${r},${c}] is misplaced`);
  });

  // Biases are one-dimensional and copied straight across.
  assert.deepEqual(
    Array.from(back.tensors.get("blk.0.ffn_down.bias").data),
    Array.from(blk.b2.d)
  );

  const params = model.paramCount();
  console.log(
    `  ${act}: ${built.tensorCount} tensors, ${built.kvCount} keys, ` +
      `${params.toLocaleString("en")} params, ${built.bytes.length} bytes`
  );
}

// ---- the optional trigram memory is carried honestly ----------------------
{
  const cfg = {
    vocab: tok.vocab,
    dim: 16,
    nLayer: 2,
    nHead: 2,
    seqLen: 12,
    ffnMult: 4,
    act: "situ",
    ngram: { size: 3, slots: 32, layer: 1 },
  };
  const model = new Transformer(cfg, mulberry32(77));
  trainStep(model, new Adam(model.params, 0.002), data, cfg.seqLen, 2, mulberry32(88));
  const built = buildModelGguf({
    model, tokenizer: tok, step: 1, loss: 3, presetName: "liten", lang: "bm",
  });
  const back = readGguf(built.bytes);
  const arch = "sprakmodell-ngram";

  assert.equal(built.arch, arch);
  assert.equal(back.kv.get("general.architecture"), arch);
  assert.equal(back.kv.get(`${arch}.ngram.size`), 3);
  assert.equal(back.kv.get(`${arch}.ngram.bucket_count`), 32);
  assert.equal(back.kv.get(`${arch}.ngram.layer`), 1);
  assert.equal(back.kv.get(`${arch}.ngram.hash`), "fnv1a-u32-token-ids-vocab-as-bos");
  assert.match(back.kv.get("general.description"), /ingen nye token/);
  assert.equal(back.kv.get("tokenizer.ggml.model"), "char");
  assert.deepEqual(back.kv.get("tokenizer.ggml.tokens"), tok.itos);

  const table = back.tensors.get("ngram_embd.weight");
  assert.ok(table, "the learned lookup table must be present");
  assert.deepEqual(table.dims, [cfg.dim, cfg.ngram.slots]);
  assert.deepEqual(Array.from(table.data), Array.from(model.ngramEmb.d));
  assert.equal(back.kv.get("sprakmodell.parameter_count"), model.paramCount());
  console.log("  trigram: lookup tensor, hash recipe and char tokenizer round-trip");
}

// ---- the writer rejects a tensor whose shape does not match its data -------

assert.throws(
  () => writeGguf([], [{ name: "bad", dims: [4, 4], data: new Float32Array(15) }]),
  /dims say 16 elements, data has 15/,
  "a mismatched tensor should be caught, not written"
);
assert.throws(
  () => writeGguf([], [{ name: "bad", dims: [0], data: new Float32Array(0) }]),
  /bad dimension/,
  "a zero dimension should be rejected"
);
console.log("  guards: mismatched and zero-sized tensors are refused");

// ---- every tensor lands exactly where the table promised -------------------
// A deliberately ragged set: none of these sizes is a multiple of the 32-byte
// alignment, so any off-by-one in the padding shows up as a wrong value.
{
  const ragged = [
    { name: "a", dims: [3], data: Float32Array.from([1, 2, 3]) },
    { name: "b", dims: [5, 2], data: Float32Array.from({ length: 10 }, (_, i) => i + 10) },
    { name: "c", dims: [7], data: Float32Array.from({ length: 7 }, (_, i) => -i) },
  ];
  const back = readGguf(writeGguf([{ key: "general.alignment", val: { t: "u32", v: 32 } }], ragged));
  for (const t of ragged)
    assert.deepEqual(
      Array.from(back.tensors.get(t.name).data),
      Array.from(t.data),
      `${t.name} should survive the padding between tensors`
    );
  console.log("  padding: three ragged tensors each land on their promised offset");
}

// ---- the download path, against a stub DOM --------------------------------
{
  const { model } = train("situ", 1);
  const built = buildModelGguf({
    model, tokenizer: tok, step: 1, loss: 4, presetName: "liten", lang: "nn",
  });

  const blob = ggufToBlob(built.bytes);
  assert.equal(blob.type, "application/octet-stream");
  assert.equal(blob.size, built.bytes.length, "the blob should hold the whole file");
  const viaBlob = new Uint8Array(await blob.arrayBuffer());
  assert.deepEqual(Array.from(viaBlob.subarray(0, 4)), [0x47, 0x47, 0x55, 0x46], "blob starts with GGUF");

  const clicks = [];
  const revoked = [];
  globalThis.URL.createObjectURL = () => "blob:stub/gguf";
  globalThis.URL.revokeObjectURL = (u) => revoked.push(u);
  const appended = [];
  globalThis.document = {
    createElement: () => ({
      click() { clicks.push({ href: this.href, download: this.download, rel: this.rel }); },
      remove() { appended.pop(); },
    }),
    body: { appendChild: (el) => appended.push(el) },
  };

  const realSetTimeout = globalThis.setTimeout;
  const parked = [];
  globalThis.setTimeout = (fn, ms, ...rest) =>
    ms >= 1000 ? (parked.push(fn), 0) : realSetTimeout(fn, ms, ...rest);
  try {
    await downloadGguf(built.bytes, "sprakmodell-liten-steg1.gguf");
  } finally {
    globalThis.setTimeout = realSetTimeout;
  }

  assert.equal(clicks.length, 1, "should click exactly one anchor");
  assert.equal(clicks[0].download, "sprakmodell-liten-steg1.gguf");
  assert.equal(clicks[0].rel, "noopener");
  assert.deepEqual(revoked, [], "must not revoke before the download starts");
  assert.equal(parked.length, 1, "should park exactly one revoke");
  parked[0]();
  assert.deepEqual(revoked, ["blob:stub/gguf"], "should revoke the object URL");
  assert.equal(appended.length, 0, "the anchor should be removed again");
  console.log("  download: anchor created, clicked with the right filename, URL revoked");
}

console.log("gguf: PASS");
