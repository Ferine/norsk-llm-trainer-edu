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
- 🧪 **Ideas from the frontier, at 1/40,000,000 scale.** Four techniques from the
  [Kimi K3 technical report](https://huggingface.co/moonshotai/Kimi-K3) are implemented for real
  and are switchable under **Flere innstillinger** — **SiTU-GLU is on by default**, because it
  costs nothing and measurably improves the model: the **Muon** optimizer with per-head
  orthogonalization, a **cosine learning-rate schedule** with warmup, the **SiTU-GLU** activation,
  and **4-bit (MXFP4) weight quantization**. See [Frontier techniques](#-frontier-techniques).

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

### 🧪 Frontier techniques

Four ideas from the **Kimi K3** technical report, ported down from 2.8T parameters to ~60k. All of
them are real implementations, not illustrations, and all are optional — the defaults are unchanged
so the guided journey stays short.

| K3 | Here | Where |
|---|---|---|
| **Per-Head Muon** (§2.5) | `Muon` in `ml.ts`. Momentum, then a quintic **Newton–Schulz** iteration that pushes every singular value of the update toward 1, so no direction of the step dominates. Q/K/V are split into one column block per head and orthogonalized separately; biases, norms, embeddings and the output head fall back to Adam. Updates are RMS-matched (`0.2·√max(dim)`) so one learning-rate slider drives both. | §5 → *Måten skruene vris på* |
| **Cosine decay + warmup** (§3.3) | `cosineLr` — 1% linear warmup, then cosine decay to 10% of the peak. K3 ran a dedicated scaling-law study to pick this over WSD. | §5 → *Bremse ned mot slutten* |
| **SiTU-GLU** (§2.3.2) | `situGlu` — `[β₁·tanh(g/β₁) ⊙ σ(g)] ⊙ [β₂·tanh(u/β₂)]` with β₁=4, β₂=25. Tracks SwiGLU near the origin but is bounded by β₁β₂ = 100, so activations cannot explode. The wide layer shrinks to ⅔ so the parameter count stays comparable to GELU and the comparison is honest. | §5 → *Knekken i det brede laget* |
| **MXFP4 quantization** (§4.1.4) | `quantizeFfnMxfp4` — blocks of 32 weights share one power-of-two scale, each weight keeps 4 bits (sign + one of `{0, ½, 1, 1½, 2, 3, 4, 6}`). About 7.5× smaller. Runs on a **clone**, so your trained model is never touched, and shows loss and generated text side by side. | §5 → *Modellen på slankekur* |

**Do they actually work at this scale?** Measured on the Bokmål corpus, preset **liten**, 3500
steps, batch 4, peak lr 8e-4, seeds fixed — held-out loss (ms/step in parentheses):

| | GELU, flat LR | GELU, cosine | SiTU-GLU, flat LR | SiTU-GLU, cosine |
|---|---|---|---|---|
| **Adam** | 0.635 *(31)* — the old default | 1.125 *(35)* | **0.363** *(31)* — **the default** | 0.370 *(30)* |
| **Muon** | 0.360 *(52)* | 0.389 *(51)* | 0.302 *(52)* | **0.264** *(53)* |

Three things worth knowing:

- **SiTU-GLU is free.** Same speed, same parameter count, 43% lower held-out loss than GELU. It is
  the single best change in the table per unit of cost.
- **Muon earns its keep,** but each step costs ~1.7× more. A learning-rate sweep at 1500 steps
  confirms the lead is not an artifact of one setting — Muon wins at every rate the slider offers
  (lr 8e-4: 1.11 vs 1.74; lr 1.5e-3: 0.62 vs 1.54; lr 3e-3: 0.55 vs 1.43).
- **The cosine schedule does not transfer.** It costs Adam+GELU dearly (0.635 → 1.125) and is
  roughly neutral elsewhere. K3 chose it for a regime where each token is seen about once; here the
  model does thousands of passes over a few paragraphs, so decaying the step size just stops it
  short. The in-app help text says so plainly rather than repeating the paper's conclusion.

Stacked, all of it is 0.635 → 0.264, a 58% reduction.

**SiTU-GLU is therefore the default.** Adam and the flat learning rate stay as they were, so the
guided workshop keeps its timing. The hero animation is always the genuine output of whatever the
defaults are, so it was regenerated for the new activation; the procedure is documented on
`Seeds.strip` in `i18n.ts` — simulate `App.tsx`'s loop (chunks of 6 steps, a sample every time the
step count passes 60, one shared sampling RNG that keeps advancing) with seeds 1337/42/7 at preset
*liten*. Getting that RNG continuity wrong reproduces step 0 correctly and every later row wrongly,
which is a confusing way to find out you regenerated it by hand.

The Excel export follows the architecture: pick SiTU-GLU and the downloaded workbook computes the
gated activation in spreadsheet formulas too (with σ written as `0.5·(1+tanh(g/2))`, since `EXP`
overflows in Excel where `TANH` never does).

### 🛑 Nothing is thrown away silently

Seven controls discard a trained model: model size, optimizer, activation, language, "rebuild with
my own text", reset, and pressing **Start** on an already-finished run. An eighth — the RLHF *reset
tuning* button — rolls the weights back to the frozen reference. All of them now route through one
confirmation dialog (`src/components/Bekreft.tsx`) that names the action, states how many steps are
at stake, and explains the consequence, in Bokmål or Nynorsk.

It only asks when there is something to lose: at step 0 the change happens immediately, and clicking
the language you are already in does nothing at all. The dialog takes Escape and a backdrop click as
"no", focuses the confirm button on open, and is an `alertdialog` for screen readers. The dropdowns
are controlled, so a cancelled change never even flickers in the UI.

`test/confirm.test.mjs` asserts the policy against the source — every call to a rebuild-triggering
setter must sit inside a `guard(...)` — because the failure mode is *adding a new switch and
forgetting*, which no runtime test would catch until a user lost their model. It also fails if
`buildEngine`'s dependency list grows a new trigger.

### 📗 Inside the exported workbook

The download (`src/lib/excel-model.ts`) is a working language model made of nothing but formulas —
no macros, no VBA. Two of its eight sheets exist to make that legible:

- **`Les_meg`** — the front page. Exactly **one cell in the whole file is editable**: the prompt,
  painted highlighter-yellow with a thick border and labelled `► SKRIV HER`. A colour legend says
  what yellow, blue and unpainted cells mean, and a test asserts the file really does contain
  exactly one such cell, so the claim cannot rot.
- **`Flytskjema`** — the pipeline as a flowchart, one boxed step per row with `▼` arrows between
  them: what happens in plain Norwegian, the formula that does it, which sheet and block it lives
  in, and **the value it is computing right now**. That last column holds live references into the
  real model, so editing the prompt and pressing F9 moves the whole chart. The final step arrows
  back to step 2 — which is exactly why the sheet needs no loop: row *t* only ever reads rows
  already computed above it.

Sixteen of the eighteen steps carry a live value; the two that don't are narrative ("this repeats
per layer") and show `—` rather than an empty cell. The test cross-checks two of those live cells
against `ml.ts` directly — the prompt, and the character the model predicts after it.

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
| `muon` | Newton–Schulz pulls every singular value into a band around 1 (checked against an independent Jacobi eigen-decomposition), is scale-invariant, splits Q/K/V per head exactly as hand-computed, covers every parameter once, and trains |
| `schedule` | Warmup ramps to the peak, cosine decays monotonically to the floor, endpoints and clamping |
| `situ` | The β₁β₂ soft cap holds for extreme inputs, SiTU-GLU tracks SwiGLU near the origin, gradients match finite differences, the ⅔ width keeps parameter counts within 5% |
| `quant` | Representable values round-trip exactly, per-block error stays under ¼ of the block maximum, byte accounting, only the wide layer is touched, the clone leaves the original alone |
| `confirm` | Every control that discards training goes through the confirmation dialog — asserted against the source, since a new switch can silently skip it and there is no React harness here to click one |
| `excel` | For **both** activations: the workbook's formulas reproduce `ml.ts` position for position, no cycles, exactly one editable cell, all 18 flowchart steps filled with live values wired to the real model, and the style table's declared counts match its contents (a mismatch is what makes Excel say "unreadable content") |

---

## 📁 Project structure

```
src/
  lib/
    ml.ts          # autograd engine, Transformer, Adam + Muon, schedules, quantization, DPO
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
    Slankekur.tsx     # 4-bit (MXFP4) quantization: size, loss and text, measured on a clone
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
