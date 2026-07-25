// ============================================================================
// Eigenskriven, liten maskinlæringsmotor (reverse-mode autograd) + transformator.
// Alt her er ekte: føreveg, tap og baklengs propagasjon (backpropagation).
// Ingenting er ferdig trent – modellen startar med tilfeldige tal.
// ============================================================================

export interface Tensor {
  d: Float32Array;
  rows: number;
  cols: number;
  grad: Float32Array;
  _prev: Tensor[];
  _back: () => void;
}

function tensor(rows: number, cols: number, prev: Tensor[] = []): Tensor {
  return {
    d: new Float32Array(rows * cols),
    rows,
    cols,
    grad: new Float32Array(rows * cols),
    _prev: prev,
    _back: () => {},
  };
}

// ---------- grunnleggjande operasjonar (kvar med si eiga deriverte) ----------

export function add(a: Tensor, b: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a, b]);
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] + b.d[i];
  out._back = () => {
    for (let i = 0; i < a.d.length; i++) {
      a.grad[i] += out.grad[i];
      b.grad[i] += out.grad[i];
    }
  };
  return out;
}

// Legg til ein rad-vektor (bias) over alle radene.
export function addRow(a: Tensor, b: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a, b]);
  const C = a.cols;
  for (let r = 0; r < a.rows; r++)
    for (let c = 0; c < C; c++) out.d[r * C + c] = a.d[r * C + c] + b.d[c];
  out._back = () => {
    for (let r = 0; r < a.rows; r++)
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        a.grad[i] += out.grad[i];
        b.grad[c] += out.grad[i];
      }
  };
  return out;
}

// Multipliser kvar kolonne med ein skalar frå rad-vektor (brukt i LayerNorm).
export function mulRow(a: Tensor, g: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a, g]);
  const C = a.cols;
  for (let r = 0; r < a.rows; r++)
    for (let c = 0; c < C; c++) out.d[r * C + c] = a.d[r * C + c] * g.d[c];
  out._back = () => {
    for (let r = 0; r < a.rows; r++)
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        a.grad[i] += out.grad[i] * g.d[c];
        g.grad[c] += out.grad[i] * a.d[i];
      }
  };
  return out;
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  const n = a.rows,
    k = a.cols,
    m = b.cols;
  const out = tensor(n, m, [a, b]);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < m; c++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a.d[r * k + p] * b.d[p * m + c];
      out.d[r * m + c] = s;
    }
  }
  out._back = () => {
    for (let r = 0; r < n; r++)
      for (let p = 0; p < k; p++) {
        let s = 0;
        for (let c = 0; c < m; c++) s += out.grad[r * m + c] * b.d[p * m + c];
        a.grad[r * k + p] += s;
      }
    for (let p = 0; p < k; p++)
      for (let c = 0; c < m; c++) {
        let s = 0;
        for (let r = 0; r < n; r++) s += a.d[r * k + p] * out.grad[r * m + c];
        b.grad[p * m + c] += s;
      }
  };
  return out;
}

export function transpose(a: Tensor): Tensor {
  const r = a.rows,
    c = a.cols;
  const out = tensor(c, r, [a]);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) out.d[j * r + i] = a.d[i * c + j];
  out._back = () => {
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++) a.grad[i * c + j] += out.grad[j * r + i];
  };
  return out;
}

export function scale(a: Tensor, s: number): Tensor {
  const out = tensor(a.rows, a.cols, [a]);
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] * s;
  out._back = () => {
    for (let i = 0; i < a.d.length; i++) a.grad[i] += out.grad[i] * s;
  };
  return out;
}

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
export function gelu(a: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a]);
  for (let i = 0; i < a.d.length; i++) {
    const x = a.d[i];
    const inner = SQRT_2_PI * (x + 0.044715 * x * x * x);
    out.d[i] = 0.5 * x * (1 + Math.tanh(inner));
  }
  out._back = () => {
    for (let i = 0; i < a.d.length; i++) {
      const x = a.d[i];
      const inner = SQRT_2_PI * (x + 0.044715 * x * x * x);
      const th = Math.tanh(inner);
      const sech2 = 1 - th * th;
      const d = 0.5 * (1 + th) + 0.5 * x * sech2 * SQRT_2_PI * (1 + 3 * 0.044715 * x * x);
      a.grad[i] += out.grad[i] * d;
    }
  };
  return out;
}

// Softmax over kvar rad (numerisk stabil).
export function softmaxRow(a: Tensor): Tensor {
  const n = a.rows,
    m = a.cols;
  const out = tensor(n, m, [a]);
  for (let r = 0; r < n; r++) {
    let mx = -Infinity;
    for (let c = 0; c < m; c++) {
      const v = a.d[r * m + c];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let c = 0; c < m; c++) {
      const e = Math.exp(a.d[r * m + c] - mx);
      out.d[r * m + c] = e;
      sum += e;
    }
    for (let c = 0; c < m; c++) out.d[r * m + c] /= sum;
  }
  out._back = () => {
    for (let r = 0; r < n; r++) {
      let s = 0;
      for (let c = 0; c < m; c++) s += out.d[r * m + c] * out.grad[r * m + c];
      for (let c = 0; c < m; c++)
        a.grad[r * m + c] += out.d[r * m + c] * (out.grad[r * m + c] - s);
    }
  };
  return out;
}

// Maskerer framtida i merksemda (kausal maske) slik at posisjon i berre
// kan sjå posisjonar <= i.
export function causalMask(a: Tensor): Tensor {
  const n = a.rows,
    m = a.cols;
  const out = tensor(n, m, [a]);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < m; c++) out.d[r * m + c] = c <= r ? a.d[r * m + c] : -1e9;
  out._back = () => {
    for (let r = 0; r < n; r++)
      for (let c = 0; c < m; c++) if (c <= r) a.grad[r * m + c] += out.grad[r * m + c];
  };
  return out;
}

// LayerNorm over den siste dimensjonen.
export function layernorm(a: Tensor, gamma: Tensor, beta: Tensor): Tensor {
  const n = a.rows,
    C = a.cols,
    eps = 1e-5;
  const out = tensor(n, C, [a, gamma, beta]);
  const xhat = new Float32Array(n * C);
  const invstd = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let mean = 0;
    for (let c = 0; c < C; c++) mean += a.d[r * C + c];
    mean /= C;
    let varr = 0;
    for (let c = 0; c < C; c++) {
      const d0 = a.d[r * C + c] - mean;
      varr += d0 * d0;
    }
    varr /= C;
    const inv = 1 / Math.sqrt(varr + eps);
    invstd[r] = inv;
    for (let c = 0; c < C; c++) {
      const xh = (a.d[r * C + c] - mean) * inv;
      xhat[r * C + c] = xh;
      out.d[r * C + c] = gamma.d[c] * xh + beta.d[c];
    }
  }
  out._back = () => {
    for (let c = 0; c < C; c++) {
      let gg = 0,
        gb = 0;
      for (let r = 0; r < n; r++) {
        const i = r * C + c;
        gg += out.grad[i] * xhat[i];
        gb += out.grad[i];
      }
      gamma.grad[c] += gg;
      beta.grad[c] += gb;
    }
    for (let r = 0; r < n; r++) {
      const inv = invstd[r];
      let sumDxhat = 0,
        sumXhatDxhat = 0;
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        const dxhat = out.grad[i] * gamma.d[c];
        sumDxhat += dxhat;
        sumXhatDxhat += xhat[i] * dxhat;
      }
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        const dxhat = out.grad[i] * gamma.d[c];
        a.grad[i] += inv * (dxhat - (sumDxhat + xhat[i] * sumXhatDxhat) / C);
      }
    }
  };
  return out;
}

export function sliceCols(a: Tensor, c0: number, c1: number): Tensor {
  const w = c1 - c0;
  const out = tensor(a.rows, w, [a]);
  for (let r = 0; r < a.rows; r++)
    for (let c = 0; c < w; c++) out.d[r * w + c] = a.d[r * a.cols + c0 + c];
  out._back = () => {
    for (let r = 0; r < a.rows; r++)
      for (let c = 0; c < w; c++) a.grad[r * a.cols + c0 + c] += out.grad[r * w + c];
  };
  return out;
}

export function concatCols(arr: Tensor[]): Tensor {
  const rows = arr[0].rows;
  let cols = 0;
  for (const a of arr) cols += a.cols;
  const out = tensor(rows, cols, arr);
  let off = 0;
  for (const a of arr) {
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < a.cols; c++) out.d[r * cols + off + c] = a.d[r * a.cols + c];
    off += a.cols;
  }
  out._back = () => {
    let off = 0;
    for (const a of arr) {
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < a.cols; c++)
          a.grad[r * a.cols + c] += out.grad[r * cols + off + c];
      off += a.cols;
    }
  };
  return out;
}

// Hentar rader frå ein embeddings-tabell (E[idx] for kvar idx).
export function gatherRows(E: Tensor, idx: number[]): Tensor {
  const T = idx.length,
    dim = E.cols;
  const out = tensor(T, dim, [E]);
  for (let r = 0; r < T; r++) {
    const e = idx[r];
    for (let c = 0; c < dim; c++) out.d[r * dim + c] = E.d[e * dim + c];
  }
  out._back = () => {
    for (let r = 0; r < T; r++) {
      const e = idx[r];
      for (let c = 0; c < dim; c++) E.grad[e * dim + c] += out.grad[r * dim + c];
    }
  };
  return out;
}

// Kryss-entropi-tap for neste-teikn-prediksjon. Logits: [T, V], mål: [T].
export function crossEntropyLoss(logits: Tensor, targets: number[]): Tensor {
  const T = logits.rows,
    V = logits.cols;
  const out = tensor(1, 1, [logits]);
  const probs = new Float32Array(T * V);
  let loss = 0;
  for (let r = 0; r < T; r++) {
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const v = logits.d[r * V + c];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let c = 0; c < V; c++) {
      const e = Math.exp(logits.d[r * V + c] - mx);
      probs[r * V + c] = e;
      sum += e;
    }
    for (let c = 0; c < V; c++) probs[r * V + c] /= sum;
    const p = probs[r * V + targets[r]];
    loss += -Math.log(p + 1e-12);
  }
  loss /= T;
  out.d[0] = loss;
  out._back = () => {
    const g0 = out.grad[0];
    for (let r = 0; r < T; r++)
      for (let c = 0; c < V; c++)
        logits.grad[r * V + c] += (g0 * (probs[r * V + c] - (c === targets[r] ? 1 : 0))) / T;
  };
  return out;
}

// Softmax of a single logits row → a probability distribution over the vocabulary.
// `pos` selects the sequence row; the result has length = vocab. Pure (no autograd).
export function rowProbs(logits: Tensor, pos: number): Float32Array {
  if (!Number.isInteger(pos) || pos < 0 || pos >= logits.rows)
    throw new RangeError(`row ${pos} is outside [0, ${logits.rows})`);
  const V = logits.cols;
  const off = pos * V;
  let mx = -Infinity;
  for (let c = 0; c < V; c++) {
    const v = logits.d[off + c];
    if (v > mx) mx = v;
  }
  let sum = 0;
  const out = new Float32Array(V);
  for (let c = 0; c < V; c++) {
    const e = Math.exp(logits.d[off + c] - mx);
    out[c] = e;
    sum += e;
  }
  for (let c = 0; c < V; c++) out[c] /= sum;
  return out;
}

// Sum of log-probabilities log softmax(logits[r0+i])[targets[i]] for i in [0, targets.length).
// Backward: d(log softmax)/d logit = onehot(target) − softmax. (autograd)
export function seqLogProb(logits: Tensor, r0: number, targets: number[]): Tensor {
  const V = logits.cols;
  const len = targets.length;
  const out = tensor(1, 1, [logits]);
  const probs = new Float32Array(len * V);
  let lp = 0;
  for (let i = 0; i < len; i++) {
    const r = r0 + i;
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const v = logits.d[r * V + c];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let c = 0; c < V; c++) {
      const e = Math.exp(logits.d[r * V + c] - mx);
      probs[i * V + c] = e;
      sum += e;
    }
    for (let c = 0; c < V; c++) probs[i * V + c] /= sum;
    lp += logits.d[r * V + targets[i]] - mx - Math.log(sum);
  }
  out.d[0] = lp;
  out._back = () => {
    const g = out.grad[0];
    for (let i = 0; i < len; i++) {
      const r = r0 + i;
      for (let c = 0; c < V; c++)
        logits.grad[r * V + c] += g * ((c === targets[i] ? 1 : 0) - probs[i * V + c]);
    }
  };
  return out;
}

// Numeric-only version (no autograd graph) for the frozen reference model.
export function seqLogProbValue(logits: Tensor, r0: number, targets: number[]): number {
  const V = logits.cols;
  let lp = 0;
  for (let i = 0; i < targets.length; i++) {
    const r = r0 + i;
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const v = logits.d[r * V + c];
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let c = 0; c < V; c++) sum += Math.exp(logits.d[r * V + c] - mx);
    lp += logits.d[r * V + targets[i]] - mx - Math.log(sum);
  }
  return lp;
}

// DPO loss for one preference pair.
// z = beta * ((lpW − refW) − (lpL − refL));  loss = softplus(−z) = −log sigmoid(z).
// d loss/dz = −sigmoid(−z);  chain: dz/dlpW = +beta, dz/dlpL = −beta.
export function dpoLoss(lpW: Tensor, lpL: Tensor, refW: number, refL: number, beta: number): Tensor {
  const out = tensor(1, 1, [lpW, lpL]);
  const z = beta * ((lpW.d[0] - refW) - (lpL.d[0] - refL));
  out.d[0] = z > 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z));
  const sigNegZ = z > 0 ? Math.exp(-z) / (1 + Math.exp(-z)) : 1 / (1 + Math.exp(z)); // sigmoid(−z)
  out._back = () => {
    const g = out.grad[0];
    const dz = -sigNegZ; // d loss/dz
    lpW.grad[0] += g * dz * beta;
    lpL.grad[0] += g * dz * -beta;
  };
  return out;
}

// Køyr baklengs propagasjon: topologisk sortering, så kall _back i omvendt rekkjefølgje.
export function backward(root: Tensor) {
  const topo: Tensor[] = [];
  const visited = new Set<Tensor>();
  // DFS gjennom grafen for å finna rekkjefølgje for baklengs propagasjon
  const build = (node: Tensor) => {
    if (visited.has(node)) return;
    visited.add(node);
    for (const p of node._prev) build(p);
    topo.push(node);
  };
  build(root);
  root.grad[0] = 1;
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._back();
}

// ---------- tilfeldig tal-generator (mulberry32) + normalfordeling ----------

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function param(rows: number, cols: number, rng: () => number, std = 0.02): Tensor {
  const d = new Float32Array(rows * cols);
  for (let i = 0; i < d.length; i++) d[i] = randn(rng) * std;
  return {
    d,
    rows,
    cols,
    grad: new Float32Array(rows * cols),
    _prev: [],
    _back: () => {},
  };
}

function zeros1(cols: number): Tensor {
  return {
    d: new Float32Array(cols),
    rows: 1,
    cols,
    grad: new Float32Array(cols),
    _prev: [],
    _back: () => {},
  };
}
function ones1(cols: number): Tensor {
  const t = zeros1(cols);
  t.d.fill(1);
  return t;
}

// ------------------------------ Transformator --------------------------------

export interface ModelConfig {
  vocab: number;
  dim: number;
  nLayer: number;
  nHead: number;
  seqLen: number;
  ffnMult: number;
}

interface Block {
  ln1g: Tensor;
  ln1b: Tensor;
  Wq: Tensor;
  Wk: Tensor;
  Wv: Tensor;
  Wo: Tensor;
  ln2g: Tensor;
  ln2b: Tensor;
  W1: Tensor;
  b1: Tensor;
  W2: Tensor;
  b2: Tensor;
}

// A single head's post-softmax attention matrix, captured for visualization.
// weights is length T*T, row-major: row = query position, col = key position.
export interface AttnView {
  layer: number;
  head: number;
  T: number;
  weights: Float32Array;
}

export class Transformer {
  cfg: ModelConfig;
  params: Tensor[];
  tokEmb: Tensor;
  posEmb: Tensor;
  blocks: Block[];
  lnFg: Tensor;
  lnFb: Tensor;
  head: Tensor;

  constructor(cfg: ModelConfig, rng: () => number) {
    if (!Number.isInteger(cfg.vocab) || cfg.vocab < 1)
      throw new RangeError("vocab must be a positive integer");
    if (!Number.isInteger(cfg.dim) || cfg.dim < 1)
      throw new RangeError("dim must be a positive integer");
    if (!Number.isInteger(cfg.nLayer) || cfg.nLayer < 1)
      throw new RangeError("nLayer must be a positive integer");
    if (!Number.isInteger(cfg.nHead) || cfg.nHead < 1 || cfg.dim % cfg.nHead !== 0)
      throw new RangeError("nHead must be a positive divisor of dim");
    if (!Number.isInteger(cfg.seqLen) || cfg.seqLen < 1)
      throw new RangeError("seqLen must be a positive integer");
    if (!Number.isInteger(cfg.ffnMult) || cfg.ffnMult < 1)
      throw new RangeError("ffnMult must be a positive integer");

    this.cfg = cfg;
    const { vocab, dim, nLayer, seqLen, ffnMult } = cfg;
    const ffn = dim * ffnMult;
    this.params = [];
    this.tokEmb = param(vocab, dim, rng, 0.02);
    this.posEmb = param(seqLen, dim, rng, 0.02);
    this.blocks = [];
    for (let i = 0; i < nLayer; i++) {
      const blk: Block = {
        ln1g: ones1(dim),
        ln1b: zeros1(dim),
        Wq: param(dim, dim, rng, 0.02),
        Wk: param(dim, dim, rng, 0.02),
        Wv: param(dim, dim, rng, 0.02),
        Wo: param(dim, dim, rng, 0.02),
        ln2g: ones1(dim),
        ln2b: zeros1(dim),
        W1: param(dim, ffn, rng, 0.02),
        b1: zeros1(ffn),
        W2: param(ffn, dim, rng, 0.02),
        b2: zeros1(dim),
      };
      this.blocks.push(blk);
    }
    this.lnFg = ones1(dim);
    this.lnFb = zeros1(dim);
    this.head = param(dim, vocab, rng, 0.02);
    for (const blk of this.blocks)
      this.params.push(
        blk.ln1g, blk.ln1b, blk.Wq, blk.Wk, blk.Wv, blk.Wo,
        blk.ln2g, blk.ln2b, blk.W1, blk.b1, blk.W2, blk.b2
      );
    this.params.push(this.tokEmb, this.posEmb, this.lnFg, this.lnFb, this.head);
  }

  get vocab() {
    return this.cfg.vocab;
  }
  get seqLen() {
    return this.cfg.seqLen;
  }

  private attention(blk: Block, x: Tensor, layer = 0, sink?: AttnView[]): Tensor {
    const q = matmul(x, blk.Wq);
    const k = matmul(x, blk.Wk);
    const v = matmul(x, blk.Wv);
    const hd = this.cfg.dim / this.cfg.nHead;
    const sc = 1 / Math.sqrt(hd);
    const heads: Tensor[] = [];
    for (let h = 0; h < this.cfg.nHead; h++) {
      const qh = sliceCols(q, h * hd, (h + 1) * hd);
      const kh = sliceCols(k, h * hd, (h + 1) * hd);
      const vh = sliceCols(v, h * hd, (h + 1) * hd);
      let scores = matmul(qh, transpose(kh));
      scores = scale(scores, sc);
      scores = causalMask(scores);
      const sm = softmaxRow(scores);
      if (sink) sink.push({ layer, head: h, T: sm.rows, weights: sm.d.slice() });
      heads.push(matmul(sm, vh));
    }
    return matmul(concatCols(heads), blk.Wo);
  }

  private ffn(blk: Block, x: Tensor): Tensor {
    let h = matmul(x, blk.W1);
    h = addRow(h, blk.b1);
    h = gelu(h);
    h = matmul(h, blk.W2);
    h = addRow(h, blk.b2);
    return h;
  }

  private blockForward(blk: Block, x: Tensor, layer = 0, sink?: AttnView[]): Tensor {
    const a = this.attention(blk, layernorm(x, blk.ln1g, blk.ln1b), layer, sink);
    x = add(x, a);
    const f = this.ffn(blk, layernorm(x, blk.ln2g, blk.ln2b));
    return add(x, f);
  }

  // Føreveg: tek token-id-ar og returnerer logits [T, vocab].
  forward(ids: number[], sink?: AttnView[]): Tensor {
    const Tt = ids.length;
    if (Tt < 1 || Tt > this.seqLen)
      throw new RangeError(`Expected between 1 and ${this.seqLen} token IDs, got ${Tt}`);
    for (const id of ids)
      if (!Number.isInteger(id) || id < 0 || id >= this.vocab)
        throw new RangeError(`Token ID ${id} is outside the vocabulary`);

    const x0 = gatherRows(this.tokEmb, ids);
    const posIdx: number[] = [];
    for (let i = 0; i < Tt; i++) posIdx[i] = i;
    let x = add(x0, gatherRows(this.posEmb, posIdx));
    for (let l = 0; l < this.blocks.length; l++) x = this.blockForward(this.blocks[l], x, l, sink);
    x = layernorm(x, this.lnFg, this.lnFb);
    return matmul(x, this.head);
  }

  // Forward pass that also records every head's post-softmax attention.
  // For visualization only — no backward pass is run on the result.
  inspect(ids: number[]): { logits: Tensor; attn: AttnView[] } {
    const attn: AttnView[] = [];
    const logits = this.forward(ids, attn);
    return { logits, attn };
  }

  paramCount(): number {
    let n = 0;
    for (const p of this.params) n += p.d.length;
    return n;
  }
}

// Deep-copy a model's parameters into a new Transformer with the same cfg.
// Used to freeze a reference policy for DPO; only forward passes run on the copy.
export function cloneTransformer(src: Transformer): Transformer {
  const dst = new Transformer(src.cfg, mulberry32(0));
  for (let i = 0; i < src.params.length; i++) dst.params[i].d.set(src.params[i].d);
  return dst;
}

// -------------------------------- Adam --------------------------------------

export class Adam {
  m: Float32Array[];
  v: Float32Array[];
  t = 0;
  constructor(
    public params: Tensor[],
    public lr = 5e-4,
    public b1 = 0.9,
    public b2 = 0.999,
    public eps = 1e-8
  ) {
    this.m = params.map((p) => new Float32Array(p.d.length));
    this.v = params.map((p) => new Float32Array(p.d.length));
  }

  zeroGrad() {
    for (const p of this.params) p.grad.fill(0);
  }

  // Klypp gradienten til ei fast norm (hindrar eksplosjon).
  clipGradNorm(maxNorm: number) {
    let total = 0;
    for (const p of this.params) for (let i = 0; i < p.grad.length; i++) total += p.grad[i] * p.grad[i];
    const norm = Math.sqrt(total);
    if (norm > maxNorm) {
      const f = maxNorm / (norm + 1e-6);
      for (const p of this.params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= f;
    }
  }

  step() {
    this.t++;
    const bc1 = 1 - Math.pow(this.b1, this.t);
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const m = this.m[i];
      const v = this.v[i];
      for (let j = 0; j < p.d.length; j++) {
        const g = p.grad[j];
        m[j] = this.b1 * m[j] + (1 - this.b1) * g;
        v[j] = this.b2 * v[j] + (1 - this.b2) * g * g;
        const mhat = m[j] / bc1;
        const vhat = v[j] / bc2;
        p.d[j] -= (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
      }
    }
  }
}

// Eitt treningssteg over ein minibatch av tilfeldige utdrag. Returnerer snitt-tap.
export function trainStep(
  model: Transformer,
  opt: Adam,
  data: number[],
  seqLen: number,
  batchSize: number,
  rng: () => number
): number {
  if (data.length < 2) throw new RangeError("Training data must contain at least two tokens");
  if (!Number.isInteger(seqLen) || seqLen < 1)
    throw new RangeError("seqLen must be a positive integer");
  if (!Number.isInteger(batchSize) || batchSize < 1)
    throw new RangeError("batchSize must be a positive integer");

  opt.zeroGrad();
  const effectiveSeqLen = Math.min(seqLen, data.length - 1, model.seqLen);
  const startCount = data.length - effectiveSeqLen;
  let total = 0;
  for (let b = 0; b < batchSize; b++) {
    const start = Math.max(0, Math.min(startCount - 1, Math.floor(rng() * startCount)));
    const x: number[] = [];
    const y: number[] = [];
    for (let i = 0; i < effectiveSeqLen; i++) {
      x.push(data[start + i]);
      y.push(data[start + i + 1]);
    }
    const logits = model.forward(x);
    const loss = crossEntropyLoss(logits, y);
    backward(loss);
    total += loss.d[0];
  }
  if (batchSize > 1)
    for (const p of model.params)
      for (let i = 0; i < p.grad.length; i++) p.grad[i] /= batchSize;
  opt.clipGradNorm(1.0);
  opt.step();
  return total / batchSize;
}

export interface PrefPair {
  promptIds: number[];
  chosenIds: number[];
  rejectedIds: number[];
}

// Build a full sequence (prompt + continuation) capped to seqLen, truncating the
// prompt from the left first so the continuation is preserved. Returns the sequence
// and P = prompt length within the cap (>= 1). Continuation = seq.slice(P) (>= 1 token).
function capSeq(promptIds: number[], contIds: number[], seqLen: number): { seq: number[]; P: number } {
  let prompt = promptIds.length ? promptIds : [0];
  let cont = contIds.slice();
  if (cont.length < 1) cont = [prompt[prompt.length - 1]];
  if (cont.length >= seqLen) cont = cont.slice(0, seqLen - 1);
  let P = prompt.length;
  if (P + cont.length > seqLen) {
    P = seqLen - cont.length;
    prompt = prompt.slice(prompt.length - P);
  }
  return { seq: prompt.concat(cont), P };
}

// One DPO update over a sampled minibatch of preference pairs.
export function dpoStep(
  policy: Transformer,
  reference: Transformer,
  opt: Adam,
  pairs: PrefPair[],
  batch: number,
  beta: number,
  rng: () => number
): { loss: number; margin: number; winRate: number } {
  if (pairs.length === 0) return { loss: 0, margin: 0, winRate: 0 };
  opt.zeroGrad();
  const seqLen = policy.seqLen;
  const n = Math.min(batch, pairs.length);
  let totalLoss = 0;
  let totalMargin = 0;
  let wins = 0;
  for (let b = 0; b < n; b++) {
    const pair = pairs[Math.min(pairs.length - 1, Math.floor(rng() * pairs.length))];
    const w = capSeq(pair.promptIds, pair.chosenIds, seqLen);
    const l = capSeq(pair.promptIds, pair.rejectedIds, seqLen);
    const tgtW = w.seq.slice(w.P);
    const tgtL = l.seq.slice(l.P);

    const lpW = seqLogProb(policy.forward(w.seq), w.P - 1, tgtW);
    const lpL = seqLogProb(policy.forward(l.seq), l.P - 1, tgtL);
    const refW = seqLogProbValue(reference.forward(w.seq), w.P - 1, tgtW);
    const refL = seqLogProbValue(reference.forward(l.seq), l.P - 1, tgtL);

    const loss = dpoLoss(lpW, lpL, refW, refL, beta);
    backward(loss);
    totalLoss += loss.d[0];
    const margin = (lpW.d[0] - refW) - (lpL.d[0] - refL);
    totalMargin += margin;
    if (margin > 0) wins++;
  }
  if (n > 1)
    for (const p of policy.params)
      for (let i = 0; i < p.grad.length; i++) p.grad[i] /= n;
  opt.clipGradNorm(1.0);
  opt.step();
  return { loss: totalLoss / n, margin: totalMargin / n, winRate: wins / n };
}

export interface SampleOpts {
  temperature: number;
  topK: number;
  length: number;
}

// Kor sikker var modellen på teiknet han valde? Full softmax over heile
// ordforrådet ved temperatur 1 – modellens eigen tru, uavhengig av kva
// temperatur og top-k brukaren har skrudd på. Difor rører ikkje
// temperatur-slideren dette talet: temperatur gjev ikkje ny kunnskap, han
// gjer berre trekkinga meir vågal, og då ser du modellen plukke teikn han
// sjølv trur lite på. Kostar O(V) per steg, V ≈ 50 – forsvinnande lite mot
// eit framoversteg.
function chosenProb(logits: Float32Array, off: number, V: number, chosen: number): number {
  let mx = -Infinity;
  for (let c = 0; c < V; c++) {
    const v = logits[off + c];
    if (v > mx) mx = v;
  }
  let sum = 0;
  for (let c = 0; c < V; c++) sum += Math.exp(logits[off + c] - mx);
  // sum >= 1 alltid (maksleddet er exp(0)), så ingen deling på null
  return Math.exp(logits[off + chosen] - mx) / sum;
}

// Sample a continuation token-by-token. Shared core for generate() and the RLHF arena.
export function sampleTokens(
  model: Transformer,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): { promptIds: number[]; contIds: number[]; conf: Float32Array } {
  let ctx = encode(prompt);
  if (ctx.length === 0) ctx = [0];
  const promptIds = ctx.slice();
  const contIds: number[] = [];
  const conf = new Float32Array(opts.length);
  const maxCtx = model.seqLen;
  const greedy = opts.temperature <= 0;
  const topK = Math.max(1, Math.min(opts.topK, model.vocab));
  for (let step = 0; step < opts.length; step++) {
    const window = ctx.length > maxCtx ? ctx.slice(ctx.length - maxCtx) : ctx;
    const logits = model.forward(window);
    const V = model.vocab;
    const off = (window.length - 1) * V;
    if (greedy) {
      let best = 0;
      let bestv = -Infinity;
      for (let c = 0; c < V; c++) {
        const val = logits.d[off + c];
        if (val > bestv) {
          bestv = val;
          best = c;
        }
      }
      conf[step] = chosenProb(logits.d, off, V, best);
      ctx.push(best);
      contIds.push(best);
      continue;
    }
    const scaled = new Float32Array(V);
    let mx = -Infinity;
    for (let c = 0; c < V; c++) {
      const val = logits.d[off + c] / opts.temperature;
      scaled[c] = val;
      if (val > mx) mx = val;
    }
    const idx: number[] = [];
    for (let c = 0; c < V; c++) idx.push(c);
    idx.sort((a, b) => scaled[b] - scaled[a]);
    const top = idx.slice(0, topK);
    let sum = 0;
    const probs = new Float32Array(top.length);
    for (let i = 0; i < top.length; i++) {
      const e = Math.exp(scaled[top[i]] - mx);
      probs[i] = e;
      sum += e;
    }
    const r = rng();
    let acc = 0;
    let chosen = top[top.length - 1];
    for (let i = 0; i < top.length; i++) {
      acc += probs[i] / sum;
      if (r <= acc) {
        chosen = top[i];
        break;
      }
    }
    conf[step] = chosenProb(logits.d, off, V, chosen);
    ctx.push(chosen);
    contIds.push(chosen);
  }
  return { promptIds, contIds, conf };
}

// Generer tekst med tal på: teksten, kvar starteksten sluttar, og kor sikker
// modellen var på kvart teikn han sjølv skreiv. Starteksten har ingen
// sikkerheit – han vart gjeven, ikkje gjetta.
export function generateDetailed(
  model: Transformer,
  decode: (ids: number[]) => string,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): { text: string; promptLen: number; conf: Float32Array } {
  const { contIds, conf } = sampleTokens(model, encode, prompt, opts, rng);
  return { text: prompt + decode(contIds), promptLen: prompt.length, conf };
}

// Generer tekst: gje ein starttekst, så lat modellen predikere teikn for teikn.
export function generate(
  model: Transformer,
  decode: (ids: number[]) => string,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): string {
  return generateDetailed(model, decode, encode, prompt, opts, rng).text;
}
