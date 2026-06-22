# Design: Bilingual UI with Bokmål default, Nynorsk optional

**Date:** 2026-06-22
**App:** In-browser character-level transformer trainer (React + Vite, single-file bundle, zero runtime deps). UI is currently Norwegian Nynorsk; the training corpus is Nynorsk.

## Goal

Make **Bokmål the default language** of the app while keeping **Nynorsk as an option**. A
language toggle flips the **entire interface** (full bilingual UI) and swaps the **training
corpus** and the **language-specific generation seeds**. Bokmål is the first thing a new visitor
sees; Nynorsk is one click away.

## Scope decisions (settled during brainstorming)

- **Full bilingual UI.** Every user-facing string has both a Bokmål and a Nynorsk variant; the
  whole interface flips with the toggle. (Not corpus-only, not labels-only.)
- **Fresh Bokmål corpus.** The Bokmål training text is written independently on similar themes
  (nature, folk tale, everyday life, proverbs, simple Q&A). It is *not* a 1:1 translation of the
  Nynorsk corpus. The existing Nynorsk corpus is retained for the Nynorsk option.
- **Default = Bokmål.** Initial language is Bokmål unless a previous choice is stored.
- **Persist the choice** in `localStorage`.
- **Reset generation seeds on switch** (prompt fields repopulate with the new language's seed).

## Non-goals

- No translation framework / i18n dependency (`react-i18next` etc.). Hand-rolled bundles only.
- No language beyond Bokmål and Nynorsk.
- No change to the ML engine (`ml.ts`), training algorithm, RLHF/DPO logic, or styling.
- No directory/repository rename (the folder stays `develop-nynorsk-llm-trainer`).

## Architecture

A single hand-rolled i18n module holds the typed language type, both string bundles, and the
per-language seeds. The corpus module gains a per-language map. `App.tsx` owns the `lang` state,
renders the toggle, resolves the active string bundle, and passes it down to the leaf components
as a prop. Switching language reuses the existing engine-rebuild path (the same one a model-size
change already triggers), so the model resets cleanly for the new corpus.

```
src/lib/i18n.ts      (new)  Lang type, LANGS metadata, Strings interface, STRINGS bundles, SEEDS
src/lib/corpus.ts    (edit) corpora: Record<Lang,string> + back-compat `corpus` export
src/App.tsx          (edit) lang state, header toggle, resolve & pass strings/seeds, rebuild wiring
src/components/Architecture.tsx (edit) takes s: Strings, no hardcoded Norwegian
src/components/Rlhf.tsx         (edit) takes s: Strings, no hardcoded Norwegian
src/components/LossChart.tsx    (edit) "Siste tap" label via prop
src/components/ui.tsx           (unchanged — structural only)
index.html           (edit) static default flips to Bokmål (lang="nb", Bokmål title)
test/i18n-parity.test.mjs (new) key-shape parity + corpus sanity
```

## Components

### 1. `src/lib/i18n.ts` (new)

```ts
export type Lang = "bm" | "nn";

export const LANGS: { id: Lang; label: string; htmlLang: string; locale: string }[] = [
  { id: "bm", label: "Bokmål",  htmlLang: "nb", locale: "nb-NO" },
  { id: "nn", label: "Nynorsk", htmlLang: "nn", locale: "nn-NO" },
];

export interface Strings {
  /* fully typed shape covering every UI string: header, hero, the seven sections’
     titles/intros, all control labels, buttons, status text, Architecture boxes &
     explanation list, Rlhf copy, LossChart label, warning note, footer. */
}

export const STRINGS: Record<Lang, Strings>;          // bm written fresh, nn = existing copy

export interface Seeds {
  chatPrompt: string;     // e.g. bm "Det var en gang" / nn "Det var ein gong"
  examples: string[];     // quick-fill buttons, language-appropriate
  sampleSentence: string; // tokenisation demo sentence
  trainSeed: string;      // seed used by the live training sampler (App.loop)
}

export const SEEDS: Record<Lang, Seeds>;
```

- The **Nynorsk** bundle is the existing copy moved verbatim out of the components — no rewording.
- The **Bokmål** bundle is newly authored, idiomatic Bokmål conveying the same meaning.
- `Strings` is a real interface so the TypeScript compiler flags any key present in one language
  but missing in the other.

### 2. `src/lib/corpus.ts` (edit)

- Add `export const corpora: Record<Lang, string>` = `{ bm: <fresh Bokmål text>, nn: <existing text> }`.
- Keep `export const corpus = corpora.bm` so the compiled test fixtures
  (`generate-parity.test.mjs`, `dpo-smoke.test.mjs`, which `import { corpus }`) keep compiling and
  running unchanged — they now exercise the Bokmål default.
- `buildTokenizer` and the `Tokenizer` interface are untouched.
- Header comment updated to describe both corpora.

### 3. `src/App.tsx` (edit)

- **State:** `const [lang, setLang] = useState<Lang>(() => readStoredLang() ?? "bm")`. A small
  `readStoredLang`/`writeStoredLang` pair wraps `localStorage["trainer-lang"]` with a guard for
  unavailable storage and invalid values.
- **Resolve:** `const s = STRINGS[lang]`, `const seed = SEEDS[lang]`, `const activeCorpus =
  corpora[lang]`.
- **Toggle:** a compact two-segment **Bokmål / Nynorsk** control in the header next to the
  subtitle. Disabled while `running || rlhf.dpoRunning` (same guard the model-size `<select>`
  uses today), so a switch can't race the training loop.
- **Rebuild on switch:** `buildEngine` uses `activeCorpus` instead of the module `corpus`, and
  `lang` joins its dependency array. The existing
  `useEffect(() => { if (!runningRef.current) buildEngine(); }, [buildEngine])` then rebuilds on
  language change exactly as it already does for a preset change: new tokenizer, model reset,
  losses cleared, `rlhf.reset()`.
- **Seed reset on switch:** an effect keyed on `lang` sets `chatPrompt` to `seed.chatPrompt` and
  `rlhf.setPrompt(seed.chatPrompt)`; `examples` and `sampleSentence` come from `seed` so they
  update automatically. The live-training sampler uses `seed.trainSeed` instead of the hardcoded
  `"Det var ein gong"`.
- **Derived display:** `displayTok`, `sampleTokens`, `vocabList`, and the `stats` fallbacks read
  from `activeCorpus` (memo deps include `lang`).
- **Locale:** `stats.params.toLocaleString("nn")` → `toLocaleString(activeLocale)` where
  `activeLocale` comes from `LANGS`.
- **Document sync:** an effect sets `document.documentElement.lang` (= `htmlLang`) and
  `document.title` (= `s.docTitle`) on language change.
- All inline Norwegian strings in App are replaced by `s.*` references.

### 4. Leaf components (edit)

- `Architecture.tsx`: add `s: Strings` prop; box titles/subs and the "Kva skjer inni?" list read
  from `s.arch.*`. Visual/SVG structure unchanged.
- `Rlhf.tsx`: add `s: Strings` prop; all copy and button labels read from `s.rlhf.*`. The
  `examples` prop already flows from App (now language-specific). Logic unchanged.
- `LossChart.tsx`: the single "Siste tap:" label comes from a prop (`label`) or `s` — passed from
  its two call sites.
- `ui.tsx` (`Section`, `Card`): structural, no literals — unchanged.

### 5. `index.html` (edit)

- `<html lang="nb">` and a Bokmål `<title>` as the static default for first paint. The runtime
  `document` effect (3) keeps them correct after a switch and on reload from a stored Nynorsk
  choice.

## Data flow

```
localStorage["trainer-lang"] ─▶ lang (useState, default "bm")
                                  │
        ┌─────────────────────────┼───────────────────────────┐
        ▼                         ▼                           ▼
  STRINGS[lang] = s        SEEDS[lang] = seed         corpora[lang] = activeCorpus
        │                         │                           │
   passed to                 prompt fields,              buildEngine(activeCorpus)
 App / Architecture /        examples, sampler          → tokenizer + model reset
 Rlhf / LossChart            (reset on switch)           (existing rebuild path)
        │                                                       │
        └──────────────── effect: document.lang/title ◀────────┘
                          effect: writeStoredLang(lang)
```

## Switch behaviour (explicit)

Switching language while **idle**: model rebuilds on the new corpus, training progress and losses
reset, RLHF state resets, prompt/example/sample text repopulate from the new seeds, document
language/title update, choice persisted. Switching while **training or DPO is running** is
prevented (toggle disabled), matching the existing model-size guard.

## Error handling

- `localStorage` access wrapped in try/catch; unavailable or invalid stored value → default `"bm"`.
- Stored value validated against known `Lang` ids before use.
- No network, no async failure surface introduced.

## Testing

- **Existing 5 tests** (`seq-logprob`, `dpo-loss`, `clone`, `generate-parity`, `dpo-smoke`) stay
  green via the back-compat `corpus = corpora.bm` export; `test:build` still compiles
  `corpus.ts` + `ml.ts` only.
- **New `test/i18n-parity.test.mjs`:** asserts (a) `STRINGS.bm` and `STRINGS.nn` have identical
  key shapes (recursively), (b) `SEEDS.bm` / `SEEDS.nn` have identical key shapes, (c)
  `corpora.bm` and `corpora.nn` are both non-empty and not equal. This catches a forgotten
  translation. `i18n.ts` is plain TypeScript with no React/JSX import, so it compiles under the
  existing `test:build` tsc invocation — that command is extended to include `src/lib/i18n.ts`,
  and the test imports `STRINGS`/`SEEDS` from `./dist/i18n.js` and `corpora` from
  `./dist/corpus.js`.
- **Manual smoke:** load → Bokmål UI + Bokmål corpus snippet; toggle → full Nynorsk UI + Nynorsk
  corpus; train a few steps in each; reload → last choice restored.

## Acceptance criteria

1. First load (no stored choice) shows a fully **Bokmål** interface and Bokmål training data.
2. A header toggle switches the **entire** UI (every section, control, button, Architecture
   diagram, RLHF panel, footer) and the corpus between Bokmål and Nynorsk.
3. Generation/training seeds (chat prompt, quick examples, tokenisation sample, training sampler)
   match the active language.
4. Switching language resets the model/training cleanly; switching is blocked mid-training.
5. The choice survives a page reload.
6. `document.documentElement.lang` and the page title reflect the active language.
7. All existing tests pass; the new parity test passes.
8. No hardcoded Norwegian UI string remains in any component (all live in `STRINGS`/`SEEDS`).
