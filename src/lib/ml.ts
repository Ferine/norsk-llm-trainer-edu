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

// Elementvis multiplikasjon (Hadamard). Brukt i porta nettverk (GLU).
export function mul(a: Tensor, b: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a, b]);
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] * b.d[i];
  out._back = () => {
    for (let i = 0; i < a.d.length; i++) {
      a.grad[i] += out.grad[i] * b.d[i];
      b.grad[i] += out.grad[i] * a.d[i];
    }
  };
  return out;
}

export function tanh(a: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a]);
  for (let i = 0; i < a.d.length; i++) out.d[i] = Math.tanh(a.d[i]);
  out._back = () => {
    for (let i = 0; i < a.d.length; i++) {
      const t = out.d[i];
      a.grad[i] += out.grad[i] * (1 - t * t);
    }
  };
  return out;
}

export function sigmoid(a: Tensor): Tensor {
  const out = tensor(a.rows, a.cols, [a]);
  for (let i = 0; i < a.d.length; i++) {
    const x = a.d[i];
    out.d[i] = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  }
  out._back = () => {
    for (let i = 0; i < a.d.length; i++) {
      const s = out.d[i];
      a.grad[i] += out.grad[i] * s * (1 - s);
    }
  };
  return out;
}

// SiTU-GLU (Sigmoid Tanh Unit GLU) frå Kimi K3, likning 12:
//   [β₁·tanh(g/β₁) ⊙ σ(g)] ⊙ [β₂·tanh(u/β₂)]
// Same form som SwiGLU nær null, men begge greinene har eit mjukt tak, så
// utdata kan aldri bli større enn β₁·β₂ = 100. Det hindrar at enkelttal
// eksploderer under trening.
export const SITU_B1 = 4;
export const SITU_B2 = 25;
export function situGlu(g: Tensor, u: Tensor): Tensor {
  const gate = mul(scale(tanh(scale(g, 1 / SITU_B1)), SITU_B1), sigmoid(g));
  const up = scale(tanh(scale(u, 1 / SITU_B2)), SITU_B2);
  return mul(gate, up);
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

// ---- Byggjeklossar for ekspertane ------------------------------------------
// Med mange små ekspertar går kvart teikn berre til nokre av dei. Då må vi
// kunna plukka ut nokre rader, rekna på dei, og leggja svaret tilbake der det
// høyrer heime. Desse tre gjer nettopp det – og har gradientar, så rutaren
// lærer kven han skal senda kva til.

// Plukkar ut eit utval rader (token). Baklengs legg gradienten tilbake på same
// rad, så eit teikn som gjekk til fleire ekspertar samlar bidraga sine.
export function takeRows(a: Tensor, idx: number[]): Tensor {
  const C = a.cols;
  const out = tensor(idx.length, C, [a]);
  for (let r = 0; r < idx.length; r++) {
    if (!Number.isInteger(idx[r]) || idx[r] < 0 || idx[r] >= a.rows)
      throw new RangeError(`Row index ${idx[r]} is outside the tensor`);
    for (let c = 0; c < C; c++) out.d[r * C + c] = a.d[idx[r] * C + c];
  }
  out._back = () => {
    for (let r = 0; r < idx.length; r++)
      for (let c = 0; c < C; c++) a.grad[idx[r] * C + c] += out.grad[r * C + c];
  };
  return out;
}

// Motstykket til takeRows: legg radene tilbake i ein større tensor, null i resten.
export function scatterRows(a: Tensor, idx: number[], rows: number): Tensor {
  const C = a.cols;
  const out = tensor(rows, C, [a]);
  for (let r = 0; r < idx.length; r++)
    for (let c = 0; c < C; c++) out.d[idx[r] * C + c] += a.d[r * C + c];
  out._back = () => {
    for (let r = 0; r < idx.length; r++)
      for (let c = 0; c < C; c++) a.grad[r * C + c] += out.grad[idx[r] * C + c];
  };
  return out;
}

// Gonger kvar rad med sin eigen skalar, henta frå ei søyle [rader, 1]. Dette
// er leddet rutaren lærer gjennom: vekta er ein tensor, ikkje ein konstant.
export function mulCol(a: Tensor, w: Tensor): Tensor {
  const C = a.cols;
  const out = tensor(a.rows, C, [a, w]);
  for (let r = 0; r < a.rows; r++)
    for (let c = 0; c < C; c++) out.d[r * C + c] = a.d[r * C + c] * w.d[r];
  out._back = () => {
    for (let r = 0; r < a.rows; r++)
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        a.grad[i] += out.grad[i] * w.d[r];
        w.grad[r] += out.grad[i] * a.d[i];
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

// Aktiveringsfunksjonen i det breie laget: den klassiske GELU, eller SiTU-GLU
// slik Kimi K3 bruker (porta, med mjukt tak).
export type Activation = "gelu" | "situ";

// Mange små ekspertar i staden for eitt breitt lag (DeepSeekMoE / DeepSeek-V3).
// Det breie laget blir delt i «experts + 1» like breie skiver: éin skiv står
// alltid på (den delte eksperten), resten blir valde av ein rutar. Talet på
// skruer er difor det same som før – berre nokre av dei er i bruk om gongen.
export interface MoeConfig {
  experts: number; // ruta ekspertar per blokk
  topK: number; // kor mange av dei eitt teikn får vekkja
  bias: number; // γ: dytten som jamnar lasta kvart steg (0 = av)
}

// Målt på bokmål-korpuset, preset «liten», 3500 steg: topK 1 og 2 gjev same
// tap (0,374 mot 0,372 – innanfor støyen), men topK 1 reknar berre 40 % av det
// breie laget og går 22 ms mot 28 ms per steg. Eitt teikn som vekkjer éin
// ekspert er dessutan det klaraste biletet i «Sjå inni modellen».
export const MOE_DEFAULT: MoeConfig = { experts: 4, topK: 1, bias: 0.001 };

// Eit lite, kontekstavhengig oppslagsminne etter ideen i Qwen3.8-Flash-Next.
// Tokeniseringa blir ikkje rørt: modellen får framleis nøyaktig éin token per
// teikn. Tre allereie eksisterande token-id-ar blir berre hasha til éi rad i
// ein ekstra tabell, og den rada blir lagd til signalet før ei vald blokk.
export interface NgramConfig {
  size: number; // kor mange teikn som dannar nøkkelen (3 = trigram)
  slots: number; // talet på rader i den hasha minnetabellen
  layer: number; // nullbasert blokk som får minnet lagt til før seg
}

export const NGRAM_DEFAULT: NgramConfig = { size: 3, slots: 256, layer: 1 };

export interface ModelConfig {
  vocab: number;
  dim: number;
  nLayer: number;
  nHead: number;
  seqLen: number;
  ffnMult: number;
  act?: Activation;
  moe?: MoeConfig;
  ngram?: NgramConfig;
}

// -1 er byrjinga på sekvensen, ikkje eit nytt token i vokabularet. Han finst
// berre medan trigramnøkkelen blir bygd, slik at dei to første teikna òg får
// kvar sin eintydige kontekst.
export const NGRAM_BOS = -1;

export function ngramKeyAt(ids: number[], pos: number, size: number): Int32Array {
  if (!Number.isInteger(size) || size < 2) throw new RangeError("ngram size must be at least 2");
  if (!Number.isInteger(pos) || pos < 0 || pos >= ids.length)
    throw new RangeError(`ngram position ${pos} is outside the sequence`);
  const key = new Int32Array(size);
  const start = pos - size + 1;
  for (let i = 0; i < size; i++) key[i] = start + i < 0 ? NGRAM_BOS : ids[start + i];
  return key;
}

// FNV-1a: lite, deterministisk og med Math.imul identisk i nettlesar og Node.
// BOS får verdien vocab; ekte token-id-ar ligg alltid i [0, vocab).
export function ngramSlot(key: ArrayLike<number>, vocab: number, slots: number): number {
  if (!Number.isInteger(vocab) || vocab < 1) throw new RangeError("vocab must be positive");
  if (!Number.isInteger(slots) || slots < 1) throw new RangeError("ngram slots must be positive");
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) {
    const raw = key[i];
    if (raw !== NGRAM_BOS && (!Number.isInteger(raw) || raw < 0 || raw >= vocab))
      throw new RangeError(`ngram token ${raw} is outside the vocabulary`);
    const id = raw === NGRAM_BOS ? vocab : raw;
    h ^= id + 1;
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) % slots;
}

export function ngramSlotsFor(ids: number[], vocab: number, cfg: NgramConfig): Int32Array {
  const out = new Int32Array(ids.length);
  for (let pos = 0; pos < ids.length; pos++)
    out[pos] = ngramSlot(ngramKeyAt(ids, pos, cfg.size), vocab, cfg.slots);
  return out;
}

// Breidda på det breie laget. GLU-varianten har tre matriser der GELU har to,
// så vi krympar breidda til 2/3 og held talet på justeringsskruer om lag likt.
// Dette er totalbreidda: med ekspertar blir ho delt, ikkje utvida.
export function ffnWidth(cfg: ModelConfig): number {
  const wide = cfg.dim * cfg.ffnMult;
  return cfg.act === "situ" ? Math.max(1, Math.round((wide * 2) / 3)) : wide;
}

// Breidda på éin ekspert: totalbreidda delt på (ruta ekspertar + den delte).
// Same rekneskap som 2/3-regelen over – vi flyttar skruer, vi legg ikkje til.
export function expertWidth(cfg: ModelConfig): number {
  if (!cfg.moe) return ffnWidth(cfg);
  return Math.max(1, Math.round(ffnWidth(cfg) / (cfg.moe.experts + 1)));
}

// Kor stor del av det breie laget som faktisk reknar for eitt teikn.
export function moeActiveFraction(cfg: ModelConfig): number {
  if (!cfg.moe) return 1;
  return (cfg.moe.topK + 1) / (cfg.moe.experts + 1);
}

// Eitt breitt lag: port-grein, opp-grein (berre SiTU-GLU) og vegen ned igjen.
export interface Expert {
  // GELU: W1/b1 er det breie laget. SiTU-GLU: W1/b1 er port-greina (W_g),
  // og Wu/bu er opp-greina (W_u).
  W1: Tensor;
  b1: Tensor;
  Wu?: Tensor;
  bu?: Tensor;
  W2: Tensor;
  b2: Tensor;
}

// Rutaren og ekspertane som står ved sida av den delte. Skeivfordelings-leddet
// `bias` er med vilje ikkje ein parameter: det styrer berre kven som blir vald,
// aldri kor mykje dei tel, og blir dytta for hand kvart steg (V3 §2.1.2).
interface Router {
  W: Tensor;
  bias: Float32Array;
  load: Float32Array;
}

interface Block extends Expert {
  ln1g: Tensor;
  ln1b: Tensor;
  Wq: Tensor;
  Wk: Tensor;
  Wv: Tensor;
  Wo: Tensor;
  ln2g: Tensor;
  ln2b: Tensor;
  // Med ekspertar er blokka sitt eige breie lag (W1/W2) den delte eksperten,
  // og `routed` er dei rutaren vel mellom.
  routed?: Expert[];
  router?: Router;
}

// Parametrane delt i to: matriser som Muon ortogonaliserer (med tal på hovud
// der matrisa er delt per hovud), og resten – bias, normaliseringar og
// tabellar – som får vanleg Adam.
export interface MuonGroups {
  matrix: { p: Tensor; heads: number }[];
  scalar: Tensor[];
}

// A single head's post-softmax attention matrix, captured for visualization.
// weights is length T*T, row-major: row = query position, col = key position.
export interface AttnView {
  layer: number;
  head: number;
  T: number;
  weights: Float32Array;
}

// Kven rutaren sende kvart teikn til, fanga for visualisering.
// gates er T*experts (rutaren si fordeling), chosen er T*topK ekspert-nummer.
export interface RouteView {
  layer: number;
  T: number;
  experts: number;
  topK: number;
  gates: Float32Array;
  chosen: Int32Array;
}

export interface NgramView {
  size: number;
  slots: number;
  layer: number;
  // T × size, med NGRAM_BOS i dei tomme plassane ved sekvensstart.
  keys: Int32Array;
  // Éin oppslagsrad per teikn. Dette er framleis ikkje token-id-ar.
  buckets: Int32Array;
}

export interface ForwardStats {
  maxActivation: number;
}

function observeActivation(a: Tensor, stats?: ForwardStats): void {
  if (!stats) return;
  for (let i = 0; i < a.d.length; i++) {
    const v = Math.abs(a.d[i]);
    if (v > stats.maxActivation) stats.maxActivation = v;
  }
}

export class Transformer {
  cfg: ModelConfig;
  params: Tensor[];
  tokEmb: Tensor;
  posEmb: Tensor;
  ngramEmb?: Tensor;
  blocks: Block[];
  lnFg: Tensor;
  lnFb: Tensor;
  head: Tensor;
  // Berre treningssteg skal telja last for rutaren; måling og generering
  // undervegs skal ikkje flytta på balansen.
  countRouting = false;

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
    if (cfg.act !== undefined && cfg.act !== "gelu" && cfg.act !== "situ")
      throw new RangeError('act must be "gelu" or "situ"');
    if (cfg.moe !== undefined) {
      const m = cfg.moe;
      if (!Number.isInteger(m.experts) || m.experts < 1)
        throw new RangeError("moe.experts must be a positive integer");
      if (!Number.isInteger(m.topK) || m.topK < 1 || m.topK > m.experts)
        throw new RangeError("moe.topK must be between 1 and moe.experts");
      if (!Number.isFinite(m.bias) || m.bias < 0)
        throw new RangeError("moe.bias must be a non-negative number");
    }
    if (cfg.ngram !== undefined) {
      const n = cfg.ngram;
      if (!Number.isInteger(n.size) || n.size < 2)
        throw new RangeError("ngram.size must be an integer of at least 2");
      if (!Number.isInteger(n.slots) || n.slots < 1)
        throw new RangeError("ngram.slots must be a positive integer");
      if (!Number.isInteger(n.layer) || n.layer < 0 || n.layer >= cfg.nLayer)
        throw new RangeError("ngram.layer must name an existing zero-based layer");
    }

    this.cfg = cfg;
    const { vocab, dim, nLayer, seqLen } = cfg;
    const situ = cfg.act === "situ";
    // Med ekspertar er kvar skive smalare; utan er eksperten heile det breie laget.
    const ffn = expertWidth(cfg);
    const mkExpert = (): Expert => {
      const e: Expert = {
        W1: param(dim, ffn, rng, 0.02),
        b1: zeros1(ffn),
        W2: param(ffn, dim, rng, 0.02),
        b2: zeros1(dim),
      };
      if (situ) {
        e.Wu = param(dim, ffn, rng, 0.02);
        e.bu = zeros1(ffn);
      }
      return e;
    };
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
        ...mkExpert(),
      };
      if (cfg.moe) {
        blk.routed = [];
        for (let e = 0; e < cfg.moe.experts; e++) blk.routed.push(mkExpert());
        blk.router = {
          W: param(dim, cfg.moe.experts, rng, 0.02),
          bias: new Float32Array(cfg.moe.experts),
          load: new Float32Array(cfg.moe.experts),
        };
      }
      this.blocks.push(blk);
    }
    this.lnFg = ones1(dim);
    this.lnFb = zeros1(dim);
    this.head = param(dim, vocab, rng, 0.02);
    const pushExpert = (e: Expert) => {
      this.params.push(e.W1, e.b1, e.W2, e.b2);
      if (e.Wu && e.bu) this.params.push(e.Wu, e.bu);
    };
    for (const blk of this.blocks) {
      this.params.push(
        blk.ln1g, blk.ln1b, blk.Wq, blk.Wk, blk.Wv, blk.Wo,
        blk.ln2g, blk.ln2b
      );
      pushExpert(blk);
      if (blk.router) this.params.push(blk.router.W);
      for (const e of blk.routed ?? []) pushExpert(e);
    }
    this.params.push(this.tokEmb, this.posEmb, this.lnFg, this.lnFb, this.head);
    // All vanlege vekter blir laga først. Dermed startar ein baseline og ein
    // trigrammodell med byte-identiske fellesvekter når dei får same frø –
    // avgjerande for ei ærleg para ablasjonsmåling.
    if (cfg.ngram) {
      this.ngramEmb = param(cfg.ngram.slots, dim, rng, 0.02);
      this.params.push(this.ngramEmb);
    }
  }

  get act(): Activation {
    return this.cfg.act ?? "gelu";
  }

  get moe(): MoeConfig | undefined {
    return this.cfg.moe;
  }

  get ngram(): NgramConfig | undefined {
    return this.cfg.ngram;
  }

  // Deler parametrane slik Muon vil ha dei: matrisene i nettverket blir
  // ortogonaliserte (spørsmål/nøkkel/verdi eitt hovud om gongen, slik K3 gjer),
  // medan tabellar, bias og normaliseringar går til Adam. Innebygging og
  // utdata-hovudet er ikkje «indre» matriser og høyrer difor til Adam.
  optimGroups(): MuonGroups {
    const matrix: { p: Tensor; heads: number }[] = [];
    const scalar: Tensor[] = [];
    const nHead = this.cfg.nHead;
    const addExpert = (e: Expert) => {
      matrix.push({ p: e.W1, heads: 1 }, { p: e.W2, heads: 1 });
      if (e.Wu) matrix.push({ p: e.Wu, heads: 1 });
      scalar.push(e.b1, e.b2);
      if (e.bu) scalar.push(e.bu);
    };
    for (const blk of this.blocks) {
      matrix.push(
        { p: blk.Wq, heads: nHead },
        { p: blk.Wk, heads: nHead },
        { p: blk.Wv, heads: nHead },
        { p: blk.Wo, heads: 1 }
      );
      addExpert(blk);
      for (const e of blk.routed ?? []) addExpert(e);
      scalar.push(blk.ln1g, blk.ln1b, blk.ln2g, blk.ln2b);
      // Rutaren er ein tabell over retningar, ikkje ei indre matrise: å presa
      // søylene hans fra kvarandre ville flytta valet, ikkje berre steget.
      // Difor Adam, same handsaming som innebygginga og utdata-hovudet.
      if (blk.router) scalar.push(blk.router.W);
    }
    scalar.push(this.tokEmb, this.posEmb, this.lnFg, this.lnFb, this.head);
    // Oppslagstabellar er ikkje lineære kart og skal difor aldri Muon-
    // ortogonaliserast. Qwen held tilsvarande n-gramtabellar på Adam.
    if (this.ngramEmb) scalar.push(this.ngramEmb);
    return { matrix, scalar };
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

  private ffn(e: Expert, x: Tensor): Tensor {
    let h: Tensor;
    if (e.Wu && e.bu) {
      // SiTU-GLU: to greiner ut i det breie laget, gonga saman.
      const g = addRow(matmul(x, e.W1), e.b1);
      const u = addRow(matmul(x, e.Wu), e.bu);
      h = situGlu(g, u);
    } else {
      h = gelu(addRow(matmul(x, e.W1), e.b1));
    }
    h = matmul(h, e.W2);
    h = addRow(h, e.b2);
    return h;
  }

  // Det breie laget som mange små. Den delte eksperten køyrer for alle teikn;
  // rutaren vel topK av dei andre per teikn og vektar svaret deira.
  private moeFfn(blk: Block, x: Tensor, layer: number, sink?: RouteView[]): Tensor {
    const cfg = this.cfg.moe!;
    const router = blk.router!;
    const routed = blk.routed!;
    const E = cfg.experts;
    const T = x.rows;

    // Rutaren fordeler éin heil porsjon merksemd over ekspertane per teikn.
    const gates = softmaxRow(matmul(x, router.W));

    // Valet blir gjort på poengsum + skeivfordelings-ledd, men vekta som blir
    // brukt er poengsummen åleine. Difor kan leddet dytta lasta jamn utan å
    // dra i gradienten – det er heile trikset i V3 §2.1.2.
    const rowsFor: number[][] = Array.from({ length: E }, () => []);
    const chosen = new Int32Array(T * cfg.topK);
    const order = new Int32Array(E);
    for (let t = 0; t < T; t++) {
      for (let e = 0; e < E; e++) order[e] = e;
      const rank = (e: number) => gates.d[t * E + e] + router.bias[e];
      // E er lite (4–8), så eit enkelt utval-sortering held.
      for (let k = 0; k < cfg.topK; k++) {
        let best = k;
        for (let j = k + 1; j < E; j++) if (rank(order[j]) > rank(order[best])) best = j;
        const tmp = order[k];
        order[k] = order[best];
        order[best] = tmp;
        const e = order[k];
        rowsFor[e].push(t);
        chosen[t * cfg.topK + k] = e;
      }
    }
    if (this.countRouting) for (let e = 0; e < E; e++) router.load[e] += rowsFor[e].length;
    if (sink)
      sink.push({ layer, T, experts: E, topK: cfg.topK, gates: gates.d.slice(), chosen });

    let out = this.ffn(blk, x);
    for (let e = 0; e < E; e++) {
      const idx = rowsFor[e];
      if (idx.length === 0) continue;
      const ye = this.ffn(routed[e], takeRows(x, idx));
      const w = takeRows(sliceCols(gates, e, e + 1), idx);
      out = add(out, scatterRows(mulCol(ye, w), idx, T));
    }
    return out;
  }

  private blockForward(
    blk: Block,
    x: Tensor,
    layer = 0,
    sink?: AttnView[],
    routeSink?: RouteView[]
  ): Tensor {
    const a = this.attention(blk, layernorm(x, blk.ln1g, blk.ln1b), layer, sink);
    x = add(x, a);
    const n = layernorm(x, blk.ln2g, blk.ln2b);
    const f = blk.router ? this.moeFfn(blk, n, layer, routeSink) : this.ffn(blk, n);
    return add(x, f);
  }

  private ngramMemory(ids: number[], sink?: NgramView): Tensor {
    const cfg = this.cfg.ngram;
    const table = this.ngramEmb;
    if (!cfg || !table) throw new Error("ngram memory is not configured");
    const buckets = ngramSlotsFor(ids, this.vocab, cfg);
    if (sink) {
      sink.size = cfg.size;
      sink.slots = cfg.slots;
      sink.layer = cfg.layer;
      sink.buckets = buckets.slice();
      sink.keys = new Int32Array(ids.length * cfg.size);
      for (let pos = 0; pos < ids.length; pos++)
        sink.keys.set(ngramKeyAt(ids, pos, cfg.size), pos * cfg.size);
    }
    return gatherRows(table, Array.from(buckets));
  }

  // Dyttar skeivfordelings-leddet mot jamn last: den som fekk meir enn sin del
  // blir litt mindre attraktiv, den som fekk mindre litt meir. Ingen hjelpe-tap,
  // ingen gradient – berre eit lite dytt, slik V3 gjer det.
  rebalanceRouters(): void {
    const gamma = this.cfg.moe?.bias ?? 0;
    for (const blk of this.blocks) {
      const r = blk.router;
      if (!r) continue;
      let total = 0;
      for (const v of r.load) total += v;
      if (total > 0 && gamma > 0) {
        const mean = total / r.load.length;
        for (let e = 0; e < r.load.length; e++) {
          if (r.load[e] > mean) r.bias[e] -= gamma;
          else if (r.load[e] < mean) r.bias[e] += gamma;
        }
      }
      r.load.fill(0);
    }
  }

  // Kor mange teikn kvar ekspert har fått sidan sist utjamning, per lag.
  routerLoad(): { bias: Float32Array; load: Float32Array }[] {
    const out: { bias: Float32Array; load: Float32Array }[] = [];
    for (const blk of this.blocks)
      if (blk.router) out.push({ bias: blk.router.bias.slice(), load: blk.router.load.slice() });
    return out;
  }

  // Føreveg: tek token-id-ar og returnerer logits [T, vocab].
  forward(
    ids: number[],
    sink?: AttnView[],
    routeSink?: RouteView[],
    ngramSink?: NgramView,
    stats?: ForwardStats
  ): Tensor {
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
    observeActivation(x, stats);
    for (let l = 0; l < this.blocks.length; l++) {
      if (this.cfg.ngram?.layer === l) {
        x = add(x, this.ngramMemory(ids, ngramSink));
        observeActivation(x, stats);
      }
      x = this.blockForward(this.blocks[l], x, l, sink, routeSink);
      observeActivation(x, stats);
    }
    x = layernorm(x, this.lnFg, this.lnFb);
    observeActivation(x, stats);
    return matmul(x, this.head);
  }

  // Forward pass that also records every head's post-softmax attention, and —
  // when the model has experts — which of them each character woke.
  // For visualization only — no backward pass is run on the result.
  inspect(ids: number[]): {
    logits: Tensor;
    attn: AttnView[];
    routes: RouteView[];
    ngram: NgramView | null;
  } {
    const attn: AttnView[] = [];
    const routes: RouteView[] = [];
    const ngram: NgramView | undefined = this.cfg.ngram
      ? { size: 0, slots: 0, layer: 0, keys: new Int32Array(), buckets: new Int32Array() }
      : undefined;
    const logits = this.forward(ids, attn, routes, ngram);
    return { logits, attn, routes, ngram: ngram ?? null };
  }

  paramCount(): number {
    let n = 0;
    for (const p of this.params) n += p.d.length;
    return n;
  }

  ngramParamCount(): number {
    return this.ngramEmb?.d.length ?? 0;
  }
}

// Deep-copy a model's parameters into a new Transformer with the same cfg.
// Used to freeze a reference policy for DPO; only forward passes run on the copy.
export function cloneTransformer(src: Transformer): Transformer {
  const dst = new Transformer(src.cfg, mulberry32(0));
  for (let i = 0; i < src.params.length; i++) dst.params[i].d.set(src.params[i].d);
  // Skeivfordelings-leddet er ingen parameter, men det avgjer kven rutaren vel.
  // Utan det ville referansemodellen i DPO rutta annleis enn den han skal måla.
  for (let i = 0; i < src.blocks.length; i++) {
    const s = src.blocks[i].router;
    const d = dst.blocks[i].router;
    if (s && d) d.bias.set(s.bias);
  }
  return dst;
}

// -------------------------------- Adam --------------------------------------

// Alt treninga treng av ein optimerar. Både Adam og Muon oppfyller han, så
// treningsløkka kan byta utan å vita kven ho snakkar med.
export interface Optimizer {
  lr: number;
  zeroGrad(): void;
  clipGradNorm(maxNorm: number): GradClipStats;
  step(): void;
}

export interface GradClipStats {
  norm: number;
  clipped: boolean;
}

// Klypp gradienten til ei fast norm (hindrar eksplosjon). Delt av alle
// optimerarane, så dei ser nøyaktig same gradient.
function clipGrads(params: Tensor[], maxNorm: number): GradClipStats {
  let total = 0;
  for (const p of params) for (let i = 0; i < p.grad.length; i++) total += p.grad[i] * p.grad[i];
  const norm = Math.sqrt(total);
  if (norm > maxNorm) {
    const f = maxNorm / (norm + 1e-6);
    for (const p of params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= f;
  }
  return { norm, clipped: norm > maxNorm };
}

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

  clipGradNorm(maxNorm: number) {
    return clipGrads(this.params, maxNorm);
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

// -------------------------------- Muon --------------------------------------
// Muon = «momentum ortogonalisert med Newton–Schulz». Adam skalerer kvart tal
// for seg; Muon ser heile matrisa under eitt og gjer retninga så jamn som mogleg
// før steget – ingen retning får dominera. Kimi K3 (§2.5) bruker han på alle
// matrisene i nettverket, og for spørsmål/nøkkel/verdi eitt hovud om gongen.

// C = A × B, der A er n×k og B er k×m. Rå Float32Array – ingen autograd.
function mmRaw(A: Float32Array, n: number, k: number, B: Float32Array, m: number): Float32Array {
  const out = new Float32Array(n * m);
  for (let r = 0; r < n; r++)
    for (let p = 0; p < k; p++) {
      const a = A[r * k + p];
      if (a === 0) continue;
      for (let c = 0; c < m; c++) out[r * m + c] += a * B[p * m + c];
    }
  return out;
}

// A × Aᵀ (n×n) for A som er n×k.
function mmTRaw(A: Float32Array, n: number, k: number): Float32Array {
  const out = new Float32Array(n * n);
  for (let r = 0; r < n; r++)
    for (let c = 0; c <= r; c++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[r * k + p] * A[c * k + p];
      out[r * n + c] = s;
      out[c * n + r] = s;
    }
  return out;
}

function transposeRaw(A: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = A[r * cols + c];
  return out;
}

// Newton–Schulz-iterasjonen: ei femtegradspolynom-oppskrift som dyttar alle
// singulærverdiane til matrisa mot 1 – altså mot ei ortogonal matrise – utan å
// rekna ein einaste singulærverdi. Fem rundar er nok i praksis.
const NS_A = 3.4445;
const NS_B = -4.775;
const NS_C = 2.0315;
export function newtonSchulz(
  G: Float32Array,
  rows: number,
  cols: number,
  steps = 5
): Float32Array {
  if (G.length !== rows * cols) throw new RangeError("newtonSchulz: shape does not match data");
  // Arbeid alltid med den låge sida ned: då er Gram-matrisa så lita som råd.
  const tall = rows > cols;
  const m = tall ? cols : rows;
  const n = tall ? rows : cols;
  let X = tall ? transposeRaw(G, rows, cols) : Float32Array.from(G);

  let fro = 0;
  for (let i = 0; i < X.length; i++) fro += X[i] * X[i];
  fro = Math.sqrt(fro);
  if (fro < 1e-12) return new Float32Array(rows * cols);
  const inv = 1 / (fro + 1e-7);
  for (let i = 0; i < X.length; i++) X[i] *= inv;

  for (let s = 0; s < steps; s++) {
    const A = mmTRaw(X, m, n); // m×m
    const AA = mmRaw(A, m, m, A, m); // m×m
    const B = new Float32Array(m * m);
    for (let i = 0; i < B.length; i++) B[i] = NS_B * A[i] + NS_C * AA[i];
    const BX = mmRaw(B, m, m, X, n); // m×n
    for (let i = 0; i < X.length; i++) X[i] = NS_A * X[i] + BX[i];
  }
  return tall ? transposeRaw(X, m, n) : X;
}

export class Muon implements Optimizer {
  params: Tensor[];
  private mom: Float32Array[];
  private adam: Adam;
  private _lr: number;

  constructor(
    public groups: MuonGroups,
    lr = 8e-4,
    public momentum = 0.95,
    public nsSteps = 5
  ) {
    this._lr = lr;
    this.mom = groups.matrix.map((g) => new Float32Array(g.p.d.length));
    this.adam = new Adam(groups.scalar, lr);
    this.params = [...groups.matrix.map((g) => g.p), ...groups.scalar];
  }

  get lr() {
    return this._lr;
  }
  set lr(v: number) {
    this._lr = v;
    this.adam.lr = v;
  }

  zeroGrad() {
    for (const p of this.params) p.grad.fill(0);
  }

  clipGradNorm(maxNorm: number) {
    return clipGrads(this.params, maxNorm);
  }

  step() {
    this.adam.step(); // bias, normaliseringar og tabellar
    for (let i = 0; i < this.groups.matrix.length; i++) {
      const { p, heads } = this.groups.matrix[i];
      const buf = this.mom[i];
      // Momentum med Nesterov-framblikk, som i Muon-referansen.
      for (let j = 0; j < buf.length; j++) buf[j] = this.momentum * buf[j] + p.grad[j];
      const rows = p.rows;
      const cols = p.cols;
      const hd = cols / heads;
      if (!Number.isInteger(hd)) throw new RangeError("Muon: heads must divide the column count");
      const blk = new Float32Array(rows * hd);
      for (let h = 0; h < heads; h++) {
        const c0 = h * hd;
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < hd; c++) {
            const j = r * cols + c0 + c;
            blk[r * hd + c] = p.grad[j] + this.momentum * buf[j];
          }
        const o = newtonSchulz(blk, rows, hd, this.nsSteps);
        // Ei ortogonal matrise har små tal (~1/√maks), så vi skalerer opp for
        // at eit Muon-steg skal bli like stort som eit Adam-steg med same lr.
        const sc = this._lr * 0.2 * Math.sqrt(Math.max(rows, hd));
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < hd; c++) p.d[r * cols + c0 + c] -= sc * o[r * hd + c];
      }
    }
  }
}

// ------------------------- læringsrate-plan (K3 §3.3) ------------------------
// Kort oppvarming (1 % av stega) og så ei kosinuskurve ned mot ein botn. K3
// gjorde eit eige skaleringsstudium og fann at dette slår ein flat rate.
export interface LrSchedule {
  peak: number;
  total: number;
  warmupFrac?: number;
  minFrac?: number;
}
export function cosineLr(step: number, o: LrSchedule): number {
  if (!(o.peak > 0)) throw new RangeError("peak learning rate must be positive");
  if (!Number.isInteger(o.total) || o.total < 1)
    throw new RangeError("total must be a positive integer");
  const warmupFrac = o.warmupFrac ?? 0.01;
  const minFrac = o.minFrac ?? 0.1;
  const min = o.peak * minFrac;
  const warm = Math.max(1, Math.floor(o.total * warmupFrac));
  const t = Math.max(0, Math.min(o.total, step));
  if (t < warm) return (o.peak * (t + 1)) / warm;
  const p = Math.min(1, (t - warm) / Math.max(1, o.total - warm));
  return min + (o.peak - min) * 0.5 * (1 + Math.cos(Math.PI * p));
}

// ---------------------- 4-bits vekter (MXFP4, K3 §4.1.4) ---------------------
// Kvart tal får berre 4 bit: eitt forteikn og ein av åtte storleikar. Kvar
// blokk på 32 tal deler ein felles todelt skala (ein toarpotens), som kostar
// eitt byte. K3 krympar berre ekspert-vektene på denne måten – her tilsvarar
// det det breie laget. Vi legg talet tilbake som float32 («fake quant»), akkurat
// som når ein trenar med kvantisering på.
// Dei åtte storleikane eit 4-bits tal kan ha. Fortegnet ligg i den fjerde biten,
// så kodane 0–7 er positive og 8–15 negative – men ±0 er same talet, så det
// finst 15 ulike verdiar, ikkje 16.
export const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];

// Den felles skalaen ei blokk får: største talet i blokka skal treffa toppen av
// skalaen (6). Eksportert så rekneark-arket kan visa nøyaktig same tal som her.
export function mxfp4Scale(maxAbs: number): number {
  if (maxAbs === 0) return 0;
  const exp = Math.max(-127, Math.min(127, Math.floor(Math.log2(maxAbs)) - 2));
  return Math.pow(2, exp);
}

// Kva av dei åtte storleikane eit tal endar på, gjeve blokka si skala.
export function mxfp4Code(v: number, scale: number): number {
  if (scale === 0) return 0;
  const a = Math.abs(v) / scale;
  let best = 0;
  let bestErr = Infinity;
  for (let c = 0; c < E2M1.length; c++) {
    const err = Math.abs(a - E2M1[c]);
    if (err < bestErr) {
      bestErr = err;
      best = c;
    }
  }
  return v < 0 ? best + 8 : best;
}

export interface QuantStats {
  values: number;
  blocks: number;
  bytesBefore: number;
  bytesAfter: number;
  meanAbsErr: number;
  maxAbsErr: number;
}

function quantizeArray(d: Float32Array, blockSize: number, acc: QuantStats) {
  for (let start = 0; start < d.length; start += blockSize) {
    const end = Math.min(d.length, start + blockSize);
    let maxAbs = 0;
    for (let i = start; i < end; i++) {
      const a = Math.abs(d[i]);
      if (a > maxAbs) maxAbs = a;
    }
    acc.blocks++;
    acc.values += end - start;
    if (maxAbs === 0) continue;
    const scale = mxfp4Scale(maxAbs);
    for (let i = start; i < end; i++) {
      const v = d[i];
      const q = Math.sign(v) * E2M1[mxfp4Code(v, scale) % 8] * scale;
      const e = Math.abs(q - v);
      acc.meanAbsErr += e;
      if (e > acc.maxAbsErr) acc.maxAbsErr = e;
      d[i] = q;
    }
  }
}

// Krympar det breie laget i alle blokkene til 4 bit, på plass. Merksemd,
// normaliseringar og tabellar står att i full presisjon, slik K3 gjer det.
export function quantizeFfnMxfp4(model: Transformer, blockSize = 32): QuantStats {
  if (!Number.isInteger(blockSize) || blockSize < 1)
    throw new RangeError("blockSize must be a positive integer");
  const acc: QuantStats = {
    values: 0,
    blocks: 0,
    bytesBefore: 0,
    bytesAfter: 0,
    meanAbsErr: 0,
    maxAbsErr: 0,
  };
  for (const blk of model.blocks) {
    // Alle ekspertane er «det breie laget» og blir krympa; rutaren er ein liten
    // tabell som avgjer vegval, og står att i full presisjon saman med
    // merksemda og normaliseringane.
    for (const e of [blk, ...(blk.routed ?? [])]) {
      quantizeArray(e.W1.d, blockSize, acc);
      if (e.Wu) quantizeArray(e.Wu.d, blockSize, acc);
      quantizeArray(e.W2.d, blockSize, acc);
    }
  }
  acc.meanAbsErr = acc.values ? acc.meanAbsErr / acc.values : 0;
  acc.bytesBefore = acc.values * 4;
  acc.bytesAfter = Math.ceil(acc.values / 2) + acc.blocks;
  return acc;
}

// Snitt-tap over nokre faste utdrag. Ingen baklengs propagasjon – berre måling.
export function evalLoss(
  model: Transformer,
  data: number[],
  seqLen: number,
  batches: number,
  rng: () => number
): number {
  if (data.length < 2) throw new RangeError("Evaluation data must contain at least two tokens");
  const effectiveSeqLen = Math.min(seqLen, data.length - 1, model.seqLen);
  const startCount = data.length - effectiveSeqLen;
  let total = 0;
  for (let b = 0; b < batches; b++) {
    const start = Math.max(0, Math.min(startCount - 1, Math.floor(rng() * startCount)));
    const x: number[] = [];
    const y: number[] = [];
    for (let i = 0; i < effectiveSeqLen; i++) {
      x.push(data[start + i]);
      y.push(data[start + i + 1]);
    }
    total += crossEntropyLoss(model.forward(x), y).d[0];
  }
  return total / batches;
}

export interface TrainStepStats {
  loss: number;
  gradNorm: number;
  clipped: boolean;
  maxActivation: number;
}

// Felles implementasjon for den raske løkka i appen og den instrumenterte
// ablasjonsløypa. Når measure=false blir ingen aktiveringar skanna.
function runTrainStep(
  model: Transformer,
  opt: Optimizer,
  data: number[],
  seqLen: number,
  batchSize: number,
  rng: () => number,
  measure: boolean
): TrainStepStats {
  if (data.length < 2) throw new RangeError("Training data must contain at least two tokens");
  if (!Number.isInteger(seqLen) || seqLen < 1)
    throw new RangeError("seqLen must be a positive integer");
  if (!Number.isInteger(batchSize) || batchSize < 1)
    throw new RangeError("batchSize must be a positive integer");

  opt.zeroGrad();
  const effectiveSeqLen = Math.min(seqLen, data.length - 1, model.seqLen);
  const startCount = data.length - effectiveSeqLen;
  let total = 0;
  const forwardStats: ForwardStats | undefined = measure ? { maxActivation: 0 } : undefined;
  model.countRouting = true;
  for (let b = 0; b < batchSize; b++) {
    const start = Math.max(0, Math.min(startCount - 1, Math.floor(rng() * startCount)));
    const x: number[] = [];
    const y: number[] = [];
    for (let i = 0; i < effectiveSeqLen; i++) {
      x.push(data[start + i]);
      y.push(data[start + i + 1]);
    }
    const logits = model.forward(x, undefined, undefined, undefined, forwardStats);
    const loss = crossEntropyLoss(logits, y);
    backward(loss);
    total += loss.d[0];
  }
  model.countRouting = false;
  if (batchSize > 1)
    for (const p of model.params)
      for (let i = 0; i < p.grad.length; i++) p.grad[i] /= batchSize;
  const clip = opt.clipGradNorm(1.0);
  opt.step();
  // Etter steget: dytt lasta jamn ut frå det denne minibatchen faktisk gjorde.
  model.rebalanceRouters();
  return {
    loss: total / batchSize,
    gradNorm: clip.norm,
    clipped: clip.clipped,
    maxActivation: forwardStats?.maxActivation ?? 0,
  };
}

// Eitt treningssteg over ein minibatch av tilfeldige utdrag. Returnerer snitt-tap.
export function trainStep(
  model: Transformer,
  opt: Optimizer,
  data: number[],
  seqLen: number,
  batchSize: number,
  rng: () => number
): number {
  return runTrainStep(model, opt, data, seqLen, batchSize, rng, false).loss;
}

// Same steg, men med dei måla ablasjonsharnessen treng for å samanlikna
// stabilitet: uklypt gradientnorm, om klypping slo inn og høgaste aktivering.
export function trainStepDetailed(
  model: Transformer,
  opt: Optimizer,
  data: number[],
  seqLen: number,
  batchSize: number,
  rng: () => number
): TrainStepStats {
  return runTrainStep(model, opt, data, seqLen, batchSize, rng, true);
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
  opt: Optimizer,
  pairs: PrefPair[],
  batch: number,
  beta: number,
  rng: () => number
): { loss: number; margin: number; winRate: number } {
  if (pairs.length === 0) return { loss: 0, margin: 0, winRate: 0 };
  // Merk: her blir ikkje skeivfordelings-leddet dytta. Under finpussinga er
  // referansemodellen frosen, og då skal rutinga liggja i ro òg – elles ville
  // dei to modellane sakte drifta frå kvarandre i kven dei spør om råd.
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

// Sample a continuation token-by-token. Shared core for generate() and the RLHF arena.
export function sampleTokens(
  model: Transformer,
  encode: (s: string) => number[],
  prompt: string,
  opts: SampleOpts,
  rng: () => number
): { promptIds: number[]; contIds: number[] } {
  let ctx = encode(prompt);
  if (ctx.length === 0) ctx = [0];
  const promptIds = ctx.slice();
  const contIds: number[] = [];
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
    ctx.push(chosen);
    contIds.push(chosen);
  }
  return { promptIds, contIds };
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
  const { contIds } = sampleTokens(model, encode, prompt, opts, rng);
  return prompt + decode(contIds);
}
