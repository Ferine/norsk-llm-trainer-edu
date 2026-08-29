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
  multi-head **causal** self-attention and a feed-forward network, with residual
  connections, a final LayerNorm, and a softmax output head. The wide layer's activation is
  switchable (SiTU-GLU by default, GELU available), and it can be split into a router plus
  experts. An optional learned **trigram memory** hashes the last three ordinary character IDs
  into one extra lookup row before block 2 — without changing the one-character/one-token
  tokenizer.
- 📉 **Real training.** Cross-entropy next-character loss, **Adam** with bias correction and
  gradient-norm clipping, minibatching — and a live loss curve so you can *see* learning happen.
- 🎛️ **Real preference tuning (RLHF).** After pretraining, fine-tune the model on *your* taste
  using **DPO** (Direct Preference Optimization) against a frozen reference copy: pick the better
  of two generated continuations and watch the preference margin and win-rate climb.
- 🔒 **100% local & private.** Builds to a **single self-contained HTML file**. No backend, no
  telemetry, no API keys — open it offline and it just works.
- 📤 **Take the model with you, two ways.** Download it as an **Excel workbook that is the model** —
  inference in plain spreadsheet formulas, no macros, no VBA — or as a **real GGUF file**, the same
  container Llama and Mistral ship in, readable by `gguf-dump` and friends. Both are written from
  scratch here, including the ZIP and GGUF writers. See
  [the workbook](#-inside-the-exported-workbook) and [the GGUF file](#-the-gguf-export).
- 🧪 **Ideas from the frontier, at 1/40,000,000 scale.** Four techniques from the
  [Kimi K3 technical report](https://huggingface.co/moonshotai/Kimi-K3) are implemented for real
  and are switchable under **Flere innstillinger** — **SiTU-GLU is on by default**, because it
  costs nothing and measurably improves the model: the **Muon** optimizer with per-head
  orthogonalization, a **cosine learning-rate schedule** with warmup, the **SiTU-GLU** activation,
  and **4-bit (MXFP4) weight quantization**. Alongside them, **DeepSeekMoE** — a mixture-of-experts
  wide layer with DeepSeek-V3's auxiliary-loss-free load balancing, where the layer is split rather
  than widened so only two fifths of it computes per character. See
  [Frontier techniques](#-frontier-techniques) and [MoE](#-mange-små-i-stedet-for-ett-stort-moe).
- 🗃️ **A Qwen-style memory you can open up.** Inspired by
  [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next), the optional 256-row trigram
  table comes with a lookup inspector (keys, buckets, learned row strength and real hash
  collisions) and a deterministic paired ablation harness that compares it against today's
  unchanged baseline. See [Trigram memory](#-trigram-memory-qwen-style).

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
pnpm run ablate:ngram # paired baseline/trigram evaluation on both corpora
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
| 1 | **What is a language model?** | The core loop: guess the next character → measure the error → nudge the weights. |
| 2 | **Raw text & tokenization** | See the Norwegian corpus, watch a sentence split into characters, and inspect the full character-level vocabulary. |
| 3 | **From characters to word-pieces (BPE)** | Learn byte-pair encoding hands-on: merge the most frequent pair step by step and watch the sample sentence re-tokenize into subwords. |
| 4 | **The architecture** | A live diagram of the transformer that updates with your chosen layer/head/dimension settings — and shows the expert split or the trigram lookup exactly where either enters the network. |
| 5 | **Training** | Pick a model size, batch size, and learning rate, then hit **Start** and watch the loss fall and sample text improve in real time. **Flere innstillinger** holds the frontier switches (optimizer, LR schedule, activation, experts, trigram memory) and the 4-bit slimming measurement. Two live weight views fold out below: the change-heat grid (*skruene*) and the **vevkart** — every single parameter as one pixel, grouped by the model's anatomy (embeddings → optional memory → per-block attention/FFN → output head), redrawn as training runs with a decaying highlighter glow on the weights that just moved. |
| 6 | **Look inside the model** | Pick any position in a sentence and inspect every head's attention pattern plus the next-character probability distribution — with the true next character as fasit. With experts switched on, an extra strip shows which expert each character woke. With trigram memory switched on, every position exposes its three-ID key, bucket, active row, learned RMS strength and other corpus trigrams colliding in that bucket. |
| 7 | **Try the model** | Give it a prompt and generate text, tuning temperature, top-k, and length. A live sliding window shows exactly which 32–48 familiar characters the model can still see before each next-character guess, while older characters fade out of reach. Because the teaching tokenizer remains one character per token, every visible character occupies exactly one context slot. The prompt you supplied stays marked in everything the model writes, so you can always see where your text ended and the model's began. |
| 8 | **RLHF** | A compact explainer based on the InstructGPT paper shows how supervised instruction tuning (SFT) turns text continuation into “instruction in, answer out.” Then generate two continuations, choose the one you prefer, and steer the model with real DPO. |
| 9 | **Add your own text** | Paste in any text to rebuild the vocabulary and retrain on your own data — or pick a bundled text from the dropdown (`src/lib/eksempeltekster.ts`): public-domain classics (Vinje, Hamsun, Garborg, Undset), CC BY-SA Wikipedia texts about language models and machine learning, and **“Trygg på nettet”**, a newly written Bokmål training text based on the factual guidance in [NEAS’ “Sikker surfing”](https://neas.no/internett/sikker-surfing/). The NEAS page states no free reuse license, so its prose is not copied: the app labels this entry as its own attributed summary. Everything ships inside the build; the app still makes zero network calls. |
| 10 | **Read more** | A friendly reading list grouped by what a learner may want next: approachable videos, visual explanations, readable code, Norwegian material, and deeper sources. Every link says what kind of resource it is and how demanding it may be; the advanced references remain available after the section is opened without dominating its introduction. Together with the GGUF visualizer in the footer, these are the app's only outbound links. |
| 11 | **Glossary** | Short, plain-language explanations of every unfamiliar term, grouped by where it appears in the learning journey. The same explanations pop up as handwritten notes when someone hovers a dotted-underlined word anywhere on the page, while source details stay with the specialist entries for readers who want them. One source (`src/lib/ordliste.ts`) feeds both views. |

Above the journey sits the **hero strip**: a real training run replayed character by character
(`LearningStrip` in `App.tsx`, data on `Seeds.strip` in `i18n.ts`). It is not an animation of what
training looks like — it is the genuine output of the shipped defaults at seeds 1337/42/7, which is
why changing a default means regenerating it.

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
`crossEntropyLoss`, the routing trio `takeRows` / `scatterRows` / `mulCol`, … — builds an output
tensor *and* wires up the closure that distributes its gradient to its inputs. Calling `backward(loss)` runs a depth-first topological sort over the
graph and invokes each `_back` in reverse, exactly like the tape in a real framework.

### The model

- **`Transformer`** — configurable `{ vocab, dim, nLayer, nHead, seqLen, ffnMult, act, moe, ngram }`, with
  validated hyperparameters, multi-head causal attention, pre-norm residual blocks, and a
  tied-shape output head. `inspect()` runs a forward pass that also records every head's attention
  every router's decisions and every hashed memory lookup, for the visualizations.
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
steps, batch 4, peak lr 8e-4, seeds fixed — loss (ms/step in parentheses). 3500 is `MAX_STEPS`, the
app's whole training budget, so these are complete runs rather than early cutoffs. The loss is
measured on fresh random windows of the corpus the model trained on, so it reports how well the
text was learned, not how it generalizes — at a few thousand characters there is no honest
generalization number to report:

| | GELU, flat LR | GELU, cosine | SiTU-GLU, flat LR | SiTU-GLU, cosine |
|---|---|---|---|---|
| **Adam** | 0.635 *(31)* — the old default | 1.125 *(35)* | **0.363** *(31)* — **the default** | 0.370 *(30)* |
| **Muon** | 0.360 *(52)* | 0.389 *(51)* | 0.302 *(52)* | **0.264** *(53)* |

Three things worth knowing:

- **SiTU-GLU is free.** Same speed, same parameter count, 43% lower loss than GELU. It is
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

### 🗃️ Trigram memory (Qwen style)

The optional **trigram memory** is inspired by
[Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next): for every position, the last
three ordinary character-token IDs (with an internal BOS sentinel only at the left edge) are
hashed with deterministic FNV-1a into one of 256 learned rows. That one `dim`-wide row is added to
the residual stream before transformer block 2. In preset *liten* this adds `256 × 48 = 12,288`
parameters, but consults only 48 of them per character.

This does **not** alter the teaching tokenizer. No trigram becomes a token, no word-piece is
inserted, and BOS is not added to the vocabulary: one Unicode character remains exactly one token.
The lookup inspector makes the distinction concrete. Click any input character to see its
three-ID key, bucket, learned row RMS, `dim / (slots × dim)` active weights, and all distinct
training-corpus trigrams that collide in the same bucket.

The switch is **off by default**, based on the paired evaluation rather than on novelty. Run it
yourself with `pnpm run ablate:ngram` (`ABLATION_STEPS`, `ABLATION_SEEDS`, `ABLATION_BATCH` and
`ABLATION_EVAL_BATCHES` can override the defaults). The harness builds both variants with the same
vocabulary and seed, verifies every shared starting weight is byte-identical, samples the same
training and held-out windows, alternates which variant runs first to balance JIT warm-up, and
records capacity, tail training loss, held-out loss, timing, gradient p99, clipping and peak
activation.

Measured on 26 August 2026: 800 steps × 3 paired seeds for each language; preset *liten*, Adam,
SiTU-GLU, batch 4, lr 8e-4, context 32. The final paragraph was held out completely.

| Language | Variant | params | tail train loss | held-out loss | ms/step | grad p99 | clip rate | max activation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bokmål | baseline (today/default) | 62,624 | 1.8019 | **2.6681** | 27.84 | 6.816 | 96.7% | 5.460 |
| Bokmål | trigram | 74,912 | **0.7998** | 3.4761 | 27.93 | 6.656 | 97.7% | 5.891 |
| Nynorsk | baseline (today/default) | 62,624 | 1.8581 | **2.7321** | 27.88 | 6.348 | 95.9% | 5.721 |
| Nynorsk | trigram | 74,912 | **0.7607** | 3.6042 | 27.87 | 6.784 | 97.0% | 5.964 |

The honest result: memory lowers tail training loss by **55.6% on Bokmål and 59.1% on Nynorsk**
at essentially zero step-time cost, while held-out loss gets **30.3% and 31.9% worse**. Stability
remains close (about +1 percentage point of clipping and +4–8% peak activation), so this is not an
explosion; it is the table doing exactly what a direct context memory is good at — memorizing this
very small corpus. That makes it a valuable ablation and teaching tool, not a quality upgrade to
turn on by default. The UI says so by leaving today's transformer selected until the learner opts
in.

#### References and implementation scope

The three primary references are also presented with bilingual reading guidance in **step 10,
Les mer**, while **step 11, Ordliste** explains n-grams, lookup tables, hash functions, collisions,
ablations, held-out data and overfitting in the vocabulary used by the inspector and benchmark.

1. **Qwen Team (2026), [*On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency,
   and Training Stability*](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf).**
   See §2.3 and Tables 7–9 (pp. 14–15): short token n-grams deterministically address embedding
   memory; one layer at Layer 2 is sufficient; larger memory lowers loss more consistently than it
   improves downstream evaluation. The report describes 51B additional table parameters held off
   the accelerator and says the table itself uses Adam without weight decay.
2. **Qwen Team (2026), [*Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate
   Cost-Efficiency*](https://qwen.ai/blog?id=qwen3.8-flash-next).** The shorter architectural
   overview explains why sparsely accessed memory can increase capacity with little per-token
   arithmetic.
3. **Qwen Team, [released `config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json).**
   This is the machine-readable check on the prose: `ngram_size` is `3`, `ple_layer_ids` is `[2]`,
   `heads_per_ngram` is `8`, and `ngram_vocab_size_base` is `20,000,000`.

“Qwen style” here therefore means **the capacity-scaling idea, trigram order, deterministic sparse
lookup and Layer-2 placement**, not a claim of architectural equivalence:

| Aspect | Qwen3.8-Flash-Next | This teaching model |
|---|---|---|
| Tokenizer | 248,320-token production vocabulary | ~70 Unicode characters; exactly one character per token |
| Memory | 51B off-accelerator table parameters | 12,288 parameters in preset *liten* |
| Retrieval | Multiple hashed heads plus key/value projections and host-memory prefetch | One transparent FNV-1a bucket and one directly added row |
| Placement | One n-gram embedding layer at Layer 2 | One trigram table before transformer block 2 |
| Purpose | Production-scale capacity and efficiency | Inspectability and a controlled generalization lesson |

FNV-1a is an implementation choice made by this project; the Qwen report specifies deterministic
addressing but does **not** prescribe FNV. Keeping that distinction explicit is why the inspector
shows the hash inputs and collisions instead of presenting the table as magic.

The spreadsheet exporter refuses a memory model until its formulas can reproduce the hash and
lookup exactly. GGUF does carry `ngram_embd.weight` plus the n-gram size, bucket count, injection
layer and hash/BOS recipe, while continuing to declare the tokenizer honestly as `char`.

### 🧩 Mange små i stedet for ett stort (MoE)

One idea from the other direction: **DeepSeekMoE**, the mixture-of-experts layer behind
[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), with its **auxiliary-loss-free load
balancing** (V3 §2.1.2). Switchable under **Flere innstillinger**, off by default.

The wide layer is **split, not widened**. With four routed experts it is cut into five equally
narrow slices: one is always on (the *shared* expert — it is simply the block's own wide layer,
narrowed), and a router picks one of the other four for each character. The parameter count is
therefore unchanged — within 2% of dense, the rounding — while only **two fifths of the wide layer
computes for any given character**. That is the actual MoE bargain, and it is the one thing a
scaled-down version can still show honestly.

The router scores every expert with a softmax, and selection happens on **score + a per-expert
bias**. The bias never enters the gate weight and never receives gradient — it is nudged by hand
each step, down for experts that took more than their share and up for the ones that took less.
That is the whole auxiliary-loss-free trick: load gets balanced without a second loss term pulling
against the real one.

Measured the same way as the table above (Bokmål corpus, preset **liten**, 3500 steps, batch 4,
lr 8e-4, seeds fixed). *Spread* is busiest expert ÷ idlest over a full sweep — the number that says
whether the router collapsed:

| | params | active | loss | ms/step | spread |
|---|---|---|---|---|---|
| **dense** — the default | 62 624 | 100% | **0.352** | 31 | — |
| 2 experts, top-1 | 63 300 | 67% | 0.371 | 26 | 1.00× |
| **4 experts, top-1** — the setting | 63 976 | **40%** | 0.374 | **22** | 1.14× |
| 4 experts, top-2 | 63 976 | 60% | 0.372 | 28 | 1.12× |
| 8 experts, top-2 | 63 576 | 33% | 0.439 | 25 | 1.25× |
| 4 experts, top-2, no balancing | 63 976 | 60% | 0.367 | 26 | 1.55× |

Four things worth knowing:

- **At this scale MoE buys speed, not quality.** Same parameters, ~6% worse loss, 29% faster steps.
  That is the honest result and the in-app help says so. The wins DeepSeek gets are wins *per
  unit of compute at scale* — here compute was never the binding constraint, a few paragraphs of
  Norwegian were.
- **Top-1 dominates top-2.** Identical loss (0.374 vs 0.372, inside the noise) for 22 ms against
  28 ms. It is also the clearest picture: one character wakes exactly one expert.
- **Fine-graining has a floor.** Eight experts means slices 14 wide, and the loss falls apart
  (0.439). There is a point below which an expert is too small to be an expert.
- **Balancing costs a little and prevents a lot.** Turning it off is slightly *better* on loss
  (0.367) and considerably worse on spread (1.55× against 1.12×) — and early in training, before
  the router settles, it was 8.5×. It exists to stop a collapse that a loss column will not show
  you until it is too late.

The Inspector gained an **expert strip**: one row per character, one column per expert, shaded by
how much of the router's attention each got, with the woken one outlined. On Norwegian text the
specialization is visible by eye — spaces, vowels and `æøå` tend to end up with different experts.

> **The workbook does not export a routed model.** `src/lib/excel-model.ts` computes one wide layer
> per block; routing would need the whole FFN section rebuilt five times over plus a new weights
> sheet. Rather than hand out a spreadsheet that computes a *different model than the one on
> screen*, `buildModelWorkbook` throws and the download button is replaced with an explanation.
> The GGUF export does carry every expert, under the architecture name `sprakmodell-moe`.

### 🛑 Nothing is thrown away silently

Nine controls discard a trained model: model size, optimizer, activation, experts, trigram memory, language,
"rebuild with my own text", reset, and pressing **Start** on an already-finished run. A tenth — the
RLHF *reset tuning* button — rolls the weights back to the frozen reference. All of them route
through one confirmation dialog (`src/components/Bekreft.tsx`) that names the action, states how
many steps are at stake, and explains the consequence, in Bokmål or Nynorsk.

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
no macros, no VBA. The `.xlsx` itself is written here too: `xlsx.ts` builds the OOXML parts and
`xlsx-zip.ts` is a from-scratch ZIP writer (stored + deflate, CRC-32, correct local and central
directory records), so nothing outside React touches the file.

The sheets are `Les_meg`, `Flytskjema`, `Vokabular`, `Vekter`, `Innebygging`, one `Lag_N` per
transformer layer, and `Utdata` — eight for the default two-layer model. A ninth, `Slankekur`,
is appended only if you ran the 4-bit measurement in the app, so the workbook carries the
quantization codes you actually looked at rather than a sheet you never asked for.

Two of them exist purely to make the rest legible:

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

### 📦 The GGUF export

The second download (`src/lib/gguf.ts`) writes the trained model as a **real GGUF v3** file — the
container Llama, Mistral and the rest are distributed in. Correct magic number, key/value metadata,
tensor table, and aligned F32 data. `gguf-dump`, `gguf-py` or any other GGUF reader will open it and
show you every screw the model learned. The writer is from scratch; no library is involved.

**It will not load in llama.cpp or ollama, and that is deliberate.** Three things differ from a
model built there, and the file states them honestly instead of papering over them:

1. **The tokenizer is character-level** (~70 characters), not byte-level BPE. llama.cpp would split
   `æ`, `ø` and `å` into bytes the model has never seen, and Norwegian would fall apart at the first
   word — so `tokenizer.ggml.model` says `char`, which is unknown to llama.cpp but true.
2. **The architecture name follows the architecture.** A GELU model genuinely *is* a gpt2 and is
   named `gpt2`. SiTU-GLU adds a gate branch gpt2 has no place for, so that file says
   `sprakmodell-situ`; a routed model says `sprakmodell-moe` and carries `expert_count`,
   `expert_used_count` and `expert_shared_count` alongside every expert's tensors and the router.
   A memory model says `sprakmodell-ngram`, carries `ngram_embd.weight`, and records the n-gram
   size, bucket count, injection layer and exact FNV-1a/BOS recipe in metadata.
3. **The attention has no bias.** gpt2 does. Rather than invent zeros to fill the slot, the tensors
   simply do not exist.

Everything else follows llama.cpp's tensor shapes and naming exactly, so the file reads as familiar
to a trained eye. Training metadata rides along in `sprakmodell.*` keys: step count, final loss,
preset, language, activation and parameter count.

The footer links to [GGUF visualizer](https://sultan-papagani.github.io/gguf-visualizer/), a
client-side 3D viewer that parses the exported file entirely in your browser — architecture,
layer/head counts, the full tensor table and a point cloud of the weights — without uploading it
anywhere. Verified against this app's own exports, custom `sprakmodell-situ` architecture included.
Its anatomy-ordered weight layout also inspired the in-app **vevkart** in step 5 — written from
scratch on 2D canvas over the live weights (`src/components/Vevkart.tsx`), no Three.js, no parsing.

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
| `i18n-parity` | Bokmål and Nynorsk string bundles have matching shapes, and every reading-list link carries a note in both languages |
| `generate-parity` | Deterministic, seed-stable generation |
| `dpo-smoke` | End-to-end: preference tuning actually moves the margin (0 → ~70, 100% win-rate) |
| `inspect` | `rowProbs` matches a manual softmax; `inspect()` logits equal `forward()`; attention is causal and each row sums to 1 |
| `bpe` | BPE learning is deterministic with correct counts/ranks/tie-breaks; encoding is monotonic in the number of merges |
| `muon` | Newton–Schulz pulls every singular value into a band around 1 (checked against an independent Jacobi eigen-decomposition), is scale-invariant, splits Q/K/V per head exactly as hand-computed, covers every parameter once, and trains |
| `schedule` | Warmup ramps to the peak, cosine decays monotonically to the floor, endpoints and clamping |
| `situ` | The β₁β₂ soft cap holds for extreme inputs, SiTU-GLU tracks SwiGLU near the origin, gradients match finite differences, the ⅔ width keeps parameter counts within 5% |
| `moe` | The three routing primitives move rows and gradients correctly (including a row routed twice), the router receives gradient, the wide layer is split rather than widened, routing is sparse and every gate row sums to 1, the balancing nudge equalizes load and resets its counters, a routed model trains without starving an expert, a clone routes identically, and the exports refuse or describe experts rather than lying about them |
| `ngram` | Unicode characters remain exactly one token each; hashing is deterministic and causal; same-seed shared weights are byte-identical; only consulted lookup rows learn; inspection, cloning and parameter accounting work; Excel refuses an unsupported memory model rather than omitting it |
| `ablation` | The paired harness preserves the character/token invariant, matches initialization and sampled windows across variants, reports finite loss/timing/gradient/clipping/activation metrics, and accounts for the table's exact capacity |
| `quant` | Representable values round-trip exactly, per-block error stays under ¼ of the block maximum, byte accounting, only the wide layer is touched, the clone leaves the original alone |
| `confirm` | Every control that discards training goes through the confirmation dialog — asserted against the source, since a new switch can silently skip it and there is no React harness here to click one |
| `excel` | For **both** activations: the workbook's formulas reproduce `ml.ts` position for position, no cycles, exactly one editable cell, all 18 flowchart steps filled with live values wired to the real model, and the style table's declared counts match its contents (a mismatch is what makes Excel say "unreadable content") |
| `xlsx-zip` | The ZIP is re-read with `node:zlib` independently of the writer, so a bad CRC, a wrong header length or a broken deflate stream surfaces here rather than in Excel; plus the browser download path |
| `gguf` | The file is parsed back by a reader written from the GGUF spec alone, not from the writer's code — wrong magic, bad alignment, a lying tensor offset or a byte-order slip fails here rather than in `gguf-dump`; both activations and the trigram table/lookup metadata round-trip, ragged tensors land on their promised offsets, and mismatched or zero-sized tensors are refused |

---

## 📁 Project structure

```
src/
  lib/
    ml.ts          # autograd engine, Transformer, Adam + Muon, schedules,
                   #   trigram lookup, quantization, MoE routing, DPO
    ablation.ts    # deterministic paired baseline-vs-trigram evaluator
    corpus.ts      # Norwegian corpora (bm/nn) + character-level tokenizer
    bpe.ts         # standalone BPE learner for the teaching demo (not the model tokenizer)
    i18n.ts        # bilingual UI strings, seeds, reading list, language metadata
    useRlhf.ts     # React hook driving the DPO preference-tuning loop
    excel-model.ts # the workbook: the model rewritten as spreadsheet formulas
    xlsx.ts        # OOXML sheet/style/workbook parts
    xlsx-zip.ts    # from-scratch ZIP writer (deflate + CRC-32) and the download path
    gguf.ts        # from-scratch GGUF v3 writer
  components/
    Architecture.tsx  # live transformer diagram
    Bekreft.tsx       # the one confirmation dialog every destructive control routes through
    BpeLab.tsx        # interactive BPE merge lab
    Inspector.tsx     # attention heatmap, probability bars, expert + lookup inspectors
    Leseliste.tsx     # the reading list — the only outbound links in the app
    LossChart.tsx     # training/DPO loss curve
    Rlhf.tsx          # preference-selection arena
    Skruer.tsx        # «Skruene vris» — live heatmap of real weight changes during training
    Slankekur.tsx     # 4-bit (MXFP4) quantization: size, loss and text, measured on a clone
    ui.tsx            # shared Section/Card/Advanced/Utskrift primitives
  utils/cn.ts      # class-name merge helper
  App.tsx          # the full guided single-page experience
  index.css        # Tailwind layer + the workbook look (paper, ink, grid)
test/              # Node test suite for the pure-logic library
scripts/
  ablate-ngram.mjs # reproducible two-language benchmark CLI
wrangler.jsonc     # Cloudflare Workers config (assets-only Worker + custom domain)
```

---

## 🛠️ Tech stack

- **React 19** + **TypeScript 5.9**
- **Vite 7** with `vite-plugin-singlefile` for a one-file build
- **Tailwind CSS 4** for styling
- **pnpm** with supply-chain hardening — exact versions (`save-exact`, empty `savePrefix`) and a
  7-day `minimumReleaseAge` cooldown, set in both `.npmrc` and `pnpm-workspace.yaml`
- **Zero ML dependencies** — the runtime packages are React, `clsx`, `tailwind-merge`, and four
  `@fontsource` families that are bundled into the build rather than fetched from a CDN, which is
  what lets the single HTML file keep its promise of no network calls

---

## ⚠️ Honest note

This is a deliberately tiny model trained in your browser on a few sentences. It is *millions of
times* smaller than production models and trains in seconds rather than weeks on vast
datasets. Expect charming nonsense, not polished prose. What's faithful is the mechanism — real
transformer, real backpropagation, real preference optimization. Feed it more text and more steps
and it gets noticeably better.

---

## 🤝 Credits

The project was put together by **z.ai GLM-5.2**, then polished and extended by
**Claude Opus 5**.

---

## 📜 License

[MIT](LICENSE) — © 2026 Atle Strand. Use it, fork it, teach with it.

The Norwegian corpus in `src/lib/corpus.ts` is original text written for this project and is
covered by the same license.
