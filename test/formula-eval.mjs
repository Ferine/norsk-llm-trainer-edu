// A very small spreadsheet engine: enough of the formula language to evaluate
// the workbook that excel-model.ts emits, so the tests can prove the sheet
// really computes the model without opening Excel.
//
// Lazy + memoized, and it throws on a reference cycle — which is exactly the
// property we most want to verify about the generated workbook.

const FUNCS = new Set([
  "SUMPRODUCT", "SUM", "MAX", "AVERAGE", "DEVSQ", "SQRT", "EXP", "TANH", "PI",
  "INDEX", "MATCH", "LEN", "MID", "UNICODE", "IF", "IFERROR", "EXACT", "ABS",
]);

// ------------------------------------------------------------------ tokenizer

const NUM = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/;
const REF = /^(?:([A-Za-z_][A-Za-z0-9_]*)!)?\$?([A-Za-z]+)\$?(\d+)(?::\$?([A-Za-z]+)\$?(\d+))?/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*/;
const OPS2 = ["<=", ">=", "<>"];
const OPS1 = "+-*/^&=<>(),";

function tokenize(src) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === " " || ch === "\t" || ch === "\n") { i++; continue; }

    if (ch === '"') {
      let j = i + 1, s = "";
      for (;;) {
        if (j >= src.length) throw new SyntaxError(`unterminated string in ${src}`);
        if (src[j] === '"') {
          if (src[j + 1] === '"') { s += '"'; j += 2; continue; }
          j++; break;
        }
        s += src[j++];
      }
      out.push({ k: "str", v: s });
      i = j;
      continue;
    }

    const rest = src.slice(i);

    const n = NUM.exec(rest);
    if (n) { out.push({ k: "num", v: Number(n[0]) }); i += n[0].length; continue; }

    // A reference always ends in digits; a function name never does, so trying
    // the reference pattern first is unambiguous.
    const r = REF.exec(rest);
    if (r) {
      out.push({ k: "ref", sheet: r[1], c1: r[2], r1: Number(r[3]), c2: r[4], r2: r[5] ? Number(r[5]) : undefined });
      i += r[0].length;
      continue;
    }

    const nm = NAME.exec(rest);
    if (nm) {
      const word = nm[0];
      i += word.length;
      if (src[i] === "(") { out.push({ k: "func", v: word.toUpperCase() }); }
      else { out.push({ k: "name", v: word }); }
      continue;
    }

    const two = rest.slice(0, 2);
    if (OPS2.includes(two)) { out.push({ k: "op", v: two }); i += 2; continue; }
    if (OPS1.includes(ch)) { out.push({ k: "op", v: ch }); i++; continue; }

    throw new SyntaxError(`unexpected ${JSON.stringify(ch)} in ${src}`);
  }
  return out;
}

function colNum(name) {
  let n = 0;
  for (const ch of name.toUpperCase()) n = n * 26 + (ch.charCodeAt(0) - 64);
  return n;
}

// -------------------------------------------------------------------- values

// A multi-cell reference evaluates to {rows, cols, v: value[][]}.
const isRange = (x) => x !== null && typeof x === "object" && Array.isArray(x.v);

function flat(x) {
  if (!isRange(x)) return [x];
  const out = [];
  for (const row of x.v) for (const cell of row) out.push(cell);
  return out;
}

function toNum(x) {
  if (isRange(x) && !(x.rows === 1 && x.cols === 1))
    throw new Error(`#VALUE! (${x.rows}x${x.cols} range used as a number)`);
  const v = isRange(x) ? x.v[0][0] : x;
  if (v === undefined || v === null || v === "") return 0;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "number") return v;
  const n = Number(v);
  if (Number.isNaN(n)) throw new Error(`#VALUE! (not a number: ${JSON.stringify(v)})`);
  return n;
}

function toStr(x) {
  const v = isRange(x) ? (x.rows === 1 && x.cols === 1 ? x.v[0][0] : "") : x;
  if (v === undefined || v === null) return "";
  if (typeof v === "boolean") return v ? "TRUE" : "FALSE";
  return String(v);
}

const nums = (x) => flat(x).filter((v) => typeof v === "number");

// ------------------------------------------------------------------- library

function callFunc(name, args) {
  switch (name) {
    case "PI": return Math.PI;
    case "SQRT": return Math.sqrt(toNum(args[0]));
    case "EXP": return Math.exp(toNum(args[0]));
    case "ABS": return Math.abs(toNum(args[0]));
    case "TANH": return Math.tanh(toNum(args[0]));
    case "LEN": return toStr(args[0]).length;
    case "UNICODE": {
      const s = toStr(args[0]);
      if (s.length === 0) throw new Error("#VALUE!");
      return s.codePointAt(0);
    }
    case "MID": {
      const s = toStr(args[0]);
      const start = Math.trunc(toNum(args[1]));
      const n = Math.trunc(toNum(args[2]));
      return s.slice(Math.max(0, start - 1), Math.max(0, start - 1) + n);
    }
    case "EXACT": return toStr(args[0]) === toStr(args[1]);
    case "SUM": return args.reduce((a, x) => a + nums(x).reduce((p, q) => p + q, 0), 0);
    case "MAX": {
      const all = args.flatMap((x) => nums(x));
      if (all.length === 0) return 0;
      return all.reduce((a, b) => (b > a ? b : a), -Infinity);
    }
    case "AVERAGE": {
      const all = args.flatMap((x) => nums(x));
      if (all.length === 0) throw new Error("#DIV/0!");
      return all.reduce((a, b) => a + b, 0) / all.length;
    }
    case "DEVSQ": {
      const all = args.flatMap((x) => nums(x));
      if (all.length === 0) throw new Error("#NUM!");
      const mean = all.reduce((a, b) => a + b, 0) / all.length;
      return all.reduce((a, b) => a + (b - mean) * (b - mean), 0);
    }
    case "SUMPRODUCT": {
      const cols = args.map((x) => flat(x));
      const n = cols[0].length;
      for (const c of cols)
        if (c.length !== n) throw new Error(`#VALUE! (SUMPRODUCT shapes ${cols.map((z) => z.length)})`);
      let acc = 0;
      for (let i = 0; i < n; i++) {
        let p = 1;
        for (const c of cols) {
          const v = c[i];
          p *= typeof v === "number" ? v : 0;
        }
        acc += p;
      }
      return acc;
    }
    case "INDEX": {
      const a = args[0];
      if (!isRange(a)) return a;
      const ri = Math.trunc(toNum(args[1]));
      if (args.length < 3) {
        // One index into a single row or column.
        const cells = flat(a);
        if (ri < 1 || ri > cells.length) throw new Error("#REF!");
        return cells[ri - 1];
      }
      const ci = Math.trunc(toNum(args[2]));
      if (ri < 1 || ri > a.rows || ci < 1 || ci > a.cols) throw new Error("#REF!");
      return a.v[ri - 1][ci - 1];
    }
    case "MATCH": {
      const want = args[0];
      const cells = flat(args[1]);
      const kind = args.length > 2 ? toNum(args[2]) : 1;
      if (kind !== 0) throw new Error("only exact MATCH is supported");
      const wn = typeof want === "number" || typeof want === "boolean" ? toNum(want) : null;
      for (let i = 0; i < cells.length; i++) {
        const c = cells[i];
        if (wn !== null) {
          if (typeof c === "number" && c === wn) return i + 1;
        } else if (typeof c === "string" && c.toUpperCase() === toStr(want).toUpperCase()) {
          // Excel's text MATCH is case-insensitive — mirrored here on purpose,
          // so the test would catch it if the workbook ever relied on case.
          return i + 1;
        }
      }
      throw new Error("#N/A");
    }
    default:
      throw new Error(`unsupported function ${name}`);
  }
}

// -------------------------------------------------------------------- engine

export function makeEngine(workbook) {
  const sheets = new Map(); // name -> Map<"r:c", cell>
  for (const sh of workbook.sheets) {
    const flatCells = new Map();
    for (const [r, line] of sh.cells) for (const [c, cell] of line) flatCells.set(`${r}:${c}`, cell);
    sheets.set(sh.name, flatCells);
  }
  const names = new Map((workbook.definedNames ?? []).map((d) => [d.name, d.ref]));

  const cache = new Map();
  const PENDING = Symbol("pending");
  let evaluated = 0;

  function rawCell(sheet, r, c) {
    const s = sheets.get(sheet);
    if (!s) throw new Error(`no sheet ${sheet}`);
    return s.get(`${r}:${c}`);
  }

  function cellValue(sheet, r, c) {
    const key = `${sheet}!${r}:${c}`;
    const hit = cache.get(key);
    if (hit === PENDING) throw new Error(`circular reference at ${key}`);
    if (hit !== undefined) return hit;
    const cell = rawCell(sheet, r, c);
    let value;
    if (cell === undefined) value = "";
    else if (cell.n !== undefined) value = cell.n;
    else if (cell.s !== undefined) value = cell.s;
    else if (cell.f !== undefined) {
      cache.set(key, PENDING);
      value = evalFormula(cell.f, sheet);
      if (isRange(value)) value = flat(value)[0];
      evaluated++;
    } else value = "";
    cache.set(key, value);
    return value;
  }

  function refValue(tok, curSheet) {
    const sheet = tok.sheet ?? curSheet;
    const c1 = colNum(tok.c1);
    const r1 = tok.r1;
    if (tok.c2 === undefined) return cellValue(sheet, r1, c1);
    const c2 = colNum(tok.c2);
    const r2 = tok.r2;
    const v = [];
    for (let r = Math.min(r1, r2); r <= Math.max(r1, r2); r++) {
      const line = [];
      for (let c = Math.min(c1, c2); c <= Math.max(c1, c2); c++) line.push(cellValue(sheet, r, c));
      v.push(line);
    }
    return { rows: v.length, cols: v[0].length, v };
  }

  function evalFormula(src, curSheet) {
    const toks = tokenize(src);
    let p = 0;
    const peek = () => toks[p];
    const eat = (v) => {
      const t = toks[p];
      if (!t || t.k !== "op" || t.v !== v) throw new SyntaxError(`expected ${v} in ${src}`);
      p++;
    };

    function primary() {
      const t = toks[p];
      if (!t) throw new SyntaxError(`unexpected end of ${src}`);
      if (t.k === "num" || t.k === "str") { p++; return t.v; }
      if (t.k === "ref") { p++; return refValue(t, curSheet); }
      if (t.k === "name") {
        p++;
        const ref = names.get(t.v);
        if (ref === undefined) throw new Error(`unknown name ${t.v}`);
        const rt = tokenize(ref)[0];
        if (!rt || rt.k !== "ref") throw new Error(`bad defined name ${t.v} -> ${ref}`);
        return refValue(rt, curSheet);
      }
      if (t.k === "func") {
        p++;
        if (!FUNCS.has(t.v)) throw new Error(`unsupported function ${t.v}`);
        eat("(");
        // Split the argument list into token spans without evaluating, so IF and
        // IFERROR can skip the branch they don't take — exactly as Excel does.
        const spans = [];
        let depth = 0;
        let start = p;
        for (;;) {
          const tk = toks[p];
          if (!tk) throw new SyntaxError(`unclosed ( in ${src}`);
          if (tk.k === "op" && tk.v === "(") depth++;
          else if (tk.k === "op" && tk.v === ")") {
            if (depth === 0) break;
            depth--;
          } else if (tk.k === "op" && tk.v === "," && depth === 0) {
            spans.push([start, p]);
            start = p + 1;
          }
          p++;
        }
        if (p > start || spans.length > 0) spans.push([start, p]);
        p++; // past ")"
        const after = p;
        const evalSpan = ([s, e]) => {
          p = s;
          const v = comparison();
          if (p !== e) throw new SyntaxError(`bad argument in ${src}`);
          p = after;
          return v;
        };

        if (t.v === "IFERROR") {
          try {
            return evalSpan(spans[0]);
          } catch (err) {
            // A cycle is a defect in the workbook, not an error value to swallow.
            if (err instanceof SyntaxError || /circular reference/.test(err.message)) throw err;
            return evalSpan(spans[1]);
          }
        }
        if (t.v === "IF") {
          if (toNum(evalSpan(spans[0])) !== 0) return evalSpan(spans[1]);
          return spans.length > 2 ? evalSpan(spans[2]) : false;
        }
        return callFunc(t.v, spans.map(evalSpan));
      }
      if (t.k === "op" && t.v === "(") { p++; const v = comparison(); eat(")"); return v; }
      if (t.k === "op" && (t.v === "-" || t.v === "+")) { p++; const v = unary(); return t.v === "-" ? -toNum(v) : toNum(v); }
      throw new SyntaxError(`unexpected ${JSON.stringify(t)} in ${src}`);
    }

    function unary() { return primary(); }

    function power() {
      let left = unary();
      while (peek() && peek().k === "op" && peek().v === "^") { p++; left = Math.pow(toNum(left), toNum(unary())); }
      return left;
    }

    function term() {
      let left = power();
      for (;;) {
        const t = peek();
        if (!t || t.k !== "op" || (t.v !== "*" && t.v !== "/")) return left;
        p++;
        const right = power();
        left = t.v === "*" ? toNum(left) * toNum(right) : toNum(left) / toNum(right);
      }
    }

    function sum() {
      let left = term();
      for (;;) {
        const t = peek();
        if (!t || t.k !== "op" || (t.v !== "+" && t.v !== "-")) return left;
        p++;
        const right = term();
        left = t.v === "+" ? toNum(left) + toNum(right) : toNum(left) - toNum(right);
      }
    }

    function concat() {
      let left = sum();
      for (;;) {
        const t = peek();
        if (!t || t.k !== "op" || t.v !== "&") return left;
        p++;
        left = toStr(left) + toStr(sum());
      }
    }

    function comparison() {
      const left = concat();
      const t = peek();
      if (!t || t.k !== "op" || !["=", "<>", "<", ">", "<=", ">="].includes(t.v)) return left;
      p++;
      const right = concat();
      const bothNum = typeof left !== "string" && typeof right !== "string";
      const a = bothNum ? toNum(left) : toStr(left).toUpperCase();
      const b = bothNum ? toNum(right) : toStr(right).toUpperCase();
      switch (t.v) {
        case "=": return a === b;
        case "<>": return a !== b;
        case "<": return a < b;
        case ">": return a > b;
        case "<=": return a <= b;
        default: return a >= b;
      }
    }

    const out = comparison();
    if (p !== toks.length) throw new SyntaxError(`trailing tokens in ${src}`);
    return out;
  }

  return {
    /** Evaluate one cell given as "Sheet!$B$4" or "Sheet!B4". */
    cell(ref) {
      const t = tokenize(ref)[0];
      if (!t || t.k !== "ref" || !t.sheet) throw new Error(`bad probe ref ${ref}`);
      return refValue(t, t.sheet);
    },
    /** Evaluate a range given as "Sheet!$B$4:$Z$4" and return a flat array. */
    range(ref) {
      const t = tokenize(ref)[0];
      if (!t || t.k !== "ref" || !t.sheet) throw new Error(`bad probe ref ${ref}`);
      return flat(refValue(t, t.sheet));
    },
    evalFormula,
    get evaluated() { return evaluated; },
  };
}
