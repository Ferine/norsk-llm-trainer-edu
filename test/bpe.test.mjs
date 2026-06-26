import { learnBpe, encodeWord, tokenizeSentence } from "./dist/bpe.js";
import assert from "node:assert/strict";

// most-frequent pair is merged first, with the right count and rank
{
  const { merges } = learnBpe("abab abab cd", 10);
  assert.equal(merges[0].a + merges[0].b, "ab", "most frequent pair 'a'+'b' merged first");
  assert.equal(merges[0].merged, "ab");
  assert.equal(merges[0].count, 4, "'a'-'b' counted 4 (2 per 'abab' x freq 2)");
  assert.equal(merges[0].rank, 0);
}

// deterministic
{
  const a = learnBpe("det er en gang det er to og det", 20);
  const b = learnBpe("det er en gang det er to og det", 20);
  assert.deepEqual(a, b, "learnBpe is deterministic");
}

// vocab grows by exactly one per merge; merged === a+b; ranks sequential
{
  const { baseVocab, merges } = learnBpe("abab abab cd", 10);
  const vocab = new Set(baseVocab);
  merges.forEach((m, i) => {
    assert.equal(m.rank, i, "ranks are sequential");
    assert.equal(m.merged, m.a + m.b, "merged === a + b");
    assert.ok(!vocab.has(m.merged), "merged token is new to the vocabulary");
    vocab.add(m.merged);
  });
  assert.equal(vocab.size, baseVocab.length + merges.length, "vocab grows by 1 per merge");
}

// ties broken lexicographically: 'a'+'b' and 'c'+'d' both occur twice → 'ab' wins
{
  const { merges } = learnBpe("ab ab cd cd", 5);
  assert.equal(merges[0].a + merges[0].b, "ab", "tie broken lexicographically → 'ab'");
}

// encodeWord: k=0 → characters; all merges → fewest tokens; monotonic non-increasing
{
  const { merges } = learnBpe("abab abab cd", 10);
  assert.deepEqual(encodeWord("abab", merges, 0), ["a", "b", "a", "b"], "k=0 → characters");
  assert.deepEqual(encodeWord("abab", merges, merges.length), ["abab"], "all merges → one token");
  let prev = Infinity;
  for (let k = 0; k <= merges.length; k++) {
    const len = encodeWord("abab", merges, k).length;
    assert.ok(len <= prev, "token count is non-increasing in k");
    prev = len;
  }
}

// tokenizeSentence: token count shrinks (or stays equal) as merges accumulate
{
  const text = "det er en gang det er to og det er";
  const { merges } = learnBpe(text, 30);
  const base = tokenizeSentence(text, merges, 0).length;
  const full = tokenizeSentence(text, merges, merges.length).length;
  assert.ok(full <= base, "token count shrinks with merges");
  assert.ok(base > 0, "base tokenization is non-empty");
}

// early stop: no pair occurs >= 2 → no merges
{
  const { merges } = learnBpe("abc", 10);
  assert.equal(merges.length, 0, "no repeated pair → no merges");
}

// empty corpus → empty vocab and no merges
{
  const { merges, baseVocab } = learnBpe("   ", 10);
  assert.equal(merges.length, 0);
  assert.deepEqual(baseVocab, []);
}

// rivals: the top-2 runner-up pairs, in lexicographic order
{
  const { merges } = learnBpe("ab ab cd cd ef ef", 1);
  assert.deepEqual(
    merges[0].rivals,
    [{ pair: "cd", count: 2 }, { pair: "ef", count: 2 }],
    "rivals are the top-2 runner-up pairs in lexicographic order"
  );
}

// encodeWord: out-of-vocabulary characters fall back to single characters
{
  const { merges } = learnBpe("abab abab cd", 10);
  assert.deepEqual(encodeWord("xyz", merges, merges.length), ["x", "y", "z"], "OOV word → characters");
  assert.deepEqual(
    encodeWord("abq", merges, merges.length),
    ["ab", "q"],
    "partial OOV → known subword + leftover character"
  );
}

console.log("bpe: PASS");
