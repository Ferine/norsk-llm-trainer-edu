# 🧠 Språkmodell-trener — train a real LLM, from scratch, in your browser

> Watch random numbers turn into Norwegian. A complete transformer language model —
> autograd, attention, Adam, and even RLHF — handwritten in TypeScript and trained live
> in front of you. No Python, no GPU, no pretrained weights, no network calls.

**Språkmodell-trener** ("language-model trainer") is an interactive, single-page web app that
teaches how modern language models actually work by letting you *build and train one yourself*.
Everything runs locally in the browser on a tiny corpus of Norwegian text, and **every part of
the machine learning is real** — the same math as ChatGPT, just very, very small.

The whole UI is bilingual: **Bokmål** (default) and **Nynorsk**, switchable with one click.

**🔗 Live demo: <https://training.aitester.win>** — nothing to install, trains in your browser.

<!-- TODO: drop a screen recording / screenshot here — e.g. ![demo](docs/demo.gif) -->

---

## ✨ Why this is different

Most "how LLMs work" demos either animate a cartoon or call out to a hosted model. This one
doesn't fake anything:

- 🔬 **A real autograd engine.** `src/lib/ml.ts` is a from-scratch reverse-mode automatic
  differentiation library — every operation (matmul, softmax, LayerNorm, GELU, attention, …)
  carries its own analytic gradient, and a topological-sort backward pass propagates them. No
  TensorFlow, no PyTorch, no `mathjs`. Just `Float32Array`s.
- 🧱 **A real transformer.** Token + positional embeddings → stacked pre-norm blocks of
  multi-head **causal** self-attention and a GELU feed-forward network, with residual
  connections, a final LayerNorm, and a softmax output head.
- 📉 **Real training.** Cross-entropy next-character loss, **Adam** with bias correction and
  gradient-norm clipping, minibatching — and a live loss curve so you can *see* learning happen.
- 🎛️ **Real preference tuning (RLHF).** After pretraining, fine-tune the model on *your* taste
  using **DPO** (Direct Preference Optimization) against a frozen reference copy: pick the better
  of two generated continuations and watch the preference margin and win-rate climb.
- 🔒 **100% local & private.** Builds to a **single self-contained HTML file**. No backend, no
  telemetry, no API keys — open it offline and it just works.

It is honest about its scale, too: this is a model with a few tens of thousands of parameters,
trained on a handful of sentences. It will not write like ChatGPT. But the *principle* is
identical, and you can watch every step of it.

---

## 🚀 Quick start

Requires **Node ≥ 20.19** and **pnpm ≥ 10**.

```bash
pnpm install      # install dependencies
pnpm dev          # start the Vite dev server (hot reload)
pnpm build        # produce a single self-contained dist/index.html
pnpm preview      # serve the production build locally
pnpm test         # run the model + i18n test suite
pnpm typecheck    # type-check without emitting
pnpm run deploy   # build + publish to Cloudflare Workers (see Deployment)
```

Then open the dev URL Vite prints (usually <http://localhost:5173>) and start training.

> The production build inlines all JS and CSS into one HTML file via
> [`vite-plugin-singlefile`](https://github.com/richardtallent/vite-plugin-singlefile) — you can
> email it, host it on any static server, or open it straight from disk.

---

## 🚢 Deployment

The site is hosted on **Cloudflare Workers** as an assets-only Worker (no server code — it just
serves the single built HTML file from the edge) at **<https://training.aitester.win>**.
Config lives in `wrangler.jsonc`: `dist/` as the assets directory, SPA fallback, and the custom
domain as a route (wrangler provisions DNS + TLS automatically).

```bash
pnpm run deploy   # vite build + wrangler deploy in one go
```

> Note the explicit `run` — plain `pnpm deploy` collides with pnpm's built-in workspace
> `deploy` command and won't do what you want.

---

## 🗺️ The guided journey

The page walks you through the full lifecycle of a language model, one section at a time:

| # | Section | What you do |
|---|---------|-------------|
| 0 | **What is a language model?** | The core loop: guess the next character → measure the error → nudge the weights. |
| 1 | **Raw text & tokenization** | See the Norwegian corpus, watch a sentence split into characters, and inspect the full character-level vocabulary. |
| 2 | **From characters to word-pieces (BPE)** | Learn byte-pair encoding hands-on: merge the most frequent pair step by step and watch the sample sentence re-tokenize into subwords. |
| 3 | **The architecture** | A live diagram of the transformer that updates with your chosen layer/head/dimension settings. |
| 4 | **Training** | Pick a model size, batch size, and learning rate, then hit **Start** and watch the loss fall and sample text improve in real time. |
| 5 | **Look inside the model** | Pick any position in a sentence and inspect every head's attention pattern plus the next-character probability distribution — with the true next character as fasit. |
| 6 | **Try the model** | Give it a prompt and generate text, tuning temperature, top-k, and length. |
| 7 | **RLHF** | Generate two continuations, choose the one you prefer, and steer the model toward your taste with DPO. |
| 8 | **Add your own text** | Paste in any text to rebuild the vocabulary and retrain on your own data. |

---

## 🔧 How it works under the hood

### The autograd core (`src/lib/ml.ts`)

Tensors are plain objects holding data, gradients, parents, and a `_back` closure:

```ts
interface Tensor {
  d: Float32Array;      // values
  grad: Float32Array;   // accumulated gradients
  rows: number; cols: number;
  _prev: Tensor[];      // inputs that produced this tensor
  _back: () => void;    // local gradient rule
}
```

Each primitive — `add`, `matmul`, `transpose`, `softmaxRow`, `layernorm`, `gelu`, `causalMask`,
`crossEntropyLoss`, … — builds an output tensor *and* wires up the closure that distributes its
gradient to its inputs. Calling `backward(loss)` runs a depth-first topological sort over the
graph and invokes each `_back` in reverse, exactly like the tape in a real framework.

### The model

- **`Transformer`** — configurable `{ vocab, dim, nLayer, nHead, seqLen, ffnMult }`, with validated
  hyperparameters, multi-head causal attention, pre-norm residual blocks, and a tied-shape output head.
- **`Adam`** — full implementation with bias-corrected moments, configurable β₁/β₂/ε, and
  `clipGradNorm` to keep gradients from exploding.
- **`trainStep`** — samples a random minibatch of windows from the corpus, runs forward → loss →
  backward → optimizer step, and returns the average loss.
- **`generate` / `sampleTokens`** — autoregressive sampling with temperature, top-k, and a greedy
  (temperature 0) path, sharing one core with the RLHF arena.

### The RLHF / DPO path

- **`cloneTransformer`** freezes a reference copy of the model.
- **`seqLogProb`** (autograd) and **`seqLogProbValue`** (numeric, for the frozen reference) score a
  continuation's log-probability.
- **`dpoLoss`** implements `−log σ(β·((logπ_w − logπ_ref,w) − (logπ_l − logπ_ref,l)))` with a
  numerically stable softplus and its exact gradient.
- **`dpoStep`** runs a preference-pair minibatch and reports loss, **margin**, and **win-rate**, all
  driven by your in-browser choices (`src/lib/useRlhf.ts`).

To stay responsive, both training and DPO run in small chunks scheduled via `setTimeout`, yielding
to the browser between bursts so the UI never freezes.

---

## 🌐 Bilingual by design

The entire interface, corpus, and example prompts exist in both **Bokmål** (`nb-NO`) and
**Nynorsk** (`nn-NO`). Strings live in `src/lib/i18n.ts`, the language choice persists to
`localStorage`, and `document.documentElement.lang` / the page title update accordingly. A test
(`i18n-parity`) enforces that both language bundles have an identical key shape so neither can
drift out of sync.

---

## 🧪 Tests

`pnpm test` compiles the pure-logic library with `tsc` and runs a Node-based suite that verifies
the math and data, independent of React:

| Test | Checks |
|------|--------|
| `seq-logprob` | `seqLogProb` autograd matches its numeric counterpart and gradients are correct |
| `dpo-loss` | DPO loss value and gradient against analytic expectations |
| `clone` | `cloneTransformer` produces an independent, identical copy |
| `corpus-lang` | Both corpora are present and non-trivial |
| `i18n-parity` | Bokmål and Nynorsk string bundles have matching shapes |
| `generate-parity` | Deterministic, seed-stable generation |
| `dpo-smoke` | End-to-end: preference tuning actually moves the margin (0 → ~70, 100% win-rate) |
| `inspect` | `rowProbs` matches a manual softmax; `inspect()` logits equal `forward()`; attention is causal and each row sums to 1 |
| `bpe` | BPE learning is deterministic with correct counts/ranks/tie-breaks; encoding is monotonic in the number of merges |

---

## 📁 Project structure

```
src/
  lib/
    ml.ts          # autograd engine, Transformer, Adam, training, generation, DPO
    corpus.ts      # Norwegian corpora (bm/nn) + character-level tokenizer
    bpe.ts         # standalone BPE learner for the teaching demo (not the model tokenizer)
    i18n.ts        # bilingual UI strings, seeds, language metadata
    useRlhf.ts     # React hook driving the DPO preference-tuning loop
  components/
    Architecture.tsx  # live transformer diagram
    BpeLab.tsx        # interactive BPE merge lab
    Inspector.tsx     # attention heatmap + next-character probability bars
    LossChart.tsx     # training/DPO loss curve
    Rlhf.tsx          # preference-selection arena
    Skruer.tsx        # «Skruene vris» — live heatmap of real weight changes during training
    ui.tsx            # shared Section/Card primitives
  App.tsx          # the full guided single-page experience
test/              # Node test suite for the pure-logic library
wrangler.jsonc     # Cloudflare Workers config (assets-only Worker + custom domain)
```

---

## 🛠️ Tech stack

- **React 19** + **TypeScript 5.9**
- **Vite 7** with `vite-plugin-singlefile` for a one-file build
- **Tailwind CSS 4** for styling
- **pnpm** with supply-chain hardening (exact versions, package release-age cooldown)
- **Zero ML dependencies** — the only runtime packages are React, `clsx`, and `tailwind-merge`

---

## ⚠️ Honest note

This is a deliberately tiny model trained in your browser on a few sentences. It is *millions of
times* smaller than production models and trains in seconds rather than weeks on vast
datasets. Expect charming nonsense, not polished prose. What's faithful is the mechanism — real
transformer, real backpropagation, real preference optimization. Feed it more text and more steps
and it gets noticeably better.

---

## 📜 License

No license file is included yet. Add one (e.g. MIT) before distributing if you intend others to
reuse the code.
