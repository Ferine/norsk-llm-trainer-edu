// ============================================================================
// Minimal .xlsx (OOXML) part-builder. Pure strings — no DOM, no zip, no deps.
// The zipping and downloading lives in xlsx-zip.ts so this file stays testable
// under plain Node.
// ============================================================================

export interface Cell {
  /** Formula without the leading "=". */
  f?: string;
  /** Numeric literal. */
  n?: number;
  /** Inline string literal. */
  s?: string;
  /** Index into the style table (see STYLE_* below). */
  st?: number;
}

export interface ColDef {
  min: number; // 1-based, inclusive
  max: number; // 1-based, inclusive
  width: number;
}

export interface SheetSpec {
  /** Must match /^[A-Za-z_][A-Za-z0-9_]*$/ so formulas never need quoting. */
  name: string;
  /** row -> col -> cell, both 1-based. */
  cells: Map<number, Map<number, Cell>>;
  cols?: ColDef[];
  /** A1-style ranges (no sheet prefix) to paint with a white→ink colour scale. */
  colorScales?: string[];
}

export interface WorkbookSpec {
  sheets: SheetSpec[];
  definedNames?: { name: string; ref: string }[];
}

// Style indices into the fixed table written by stylesXml().
export const STYLE_DEFAULT = 0;
export const STYLE_BOLD = 1;
export const STYLE_WRAP = 2;
export const STYLE_MONO = 3;
/** Den eine cella brukaren skal skriva i: gul som ein tusj, tjukk ramme. */
export const STYLE_INPUT = 4;
/** Eit svar rekna ut av modellen. */
export const STYLE_RESULT = 5;
/** Ein boks i flytskjemaet. */
export const STYLE_BOX = 6;
/** Overskrifta i ein slik boks. */
export const STYLE_STEP = 7;
/** Formelen i ein slik boks, med fast breidd. */
export const STYLE_CODE = 8;

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function newSheet(name: string): SheetSpec {
  if (!NAME_RE.test(name) || name.length > 31)
    throw new RangeError(`unsafe sheet name: ${name}`);
  return { name, cells: new Map() };
}

export function put(sh: SheetSpec, row: number, col: number, cell: Cell): void {
  let r = sh.cells.get(row);
  if (!r) {
    r = new Map();
    sh.cells.set(row, r);
  }
  r.set(col, cell);
}

/** 1-based column number -> "A", "B", ... "AA". */
export function colName(col: number): string {
  let n = col;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = (n - 1 - rem) / 26;
  }
  return out;
}

export function esc(s: string): string {
  let out = "";
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (ch === "&") out += "&amp;";
    else if (ch === "<") out += "&lt;";
    else if (ch === ">") out += "&gt;";
    else if (ch === '"') out += "&quot;";
    // Control characters go out as numeric references — tab/LF/CR because a raw
    // newline inside <t> is fragile, the rest because XML 1.0 forbids them.
    else if (c < 0x20) out += `&#${c};`;
    else out += ch;
  }
  return out;
}

/** A double in a form Excel's XML parser is happy with (uppercase exponent). */
export function num(v: number): string {
  if (!Number.isFinite(v)) return "0";
  if (Object.is(v, -0)) return "0";
  return String(v).replace("e", "E");
}

const HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>';
const NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

function cellXml(row: number, col: number, cell: Cell): string {
  const ref = `${colName(col)}${row}`;
  const st = cell.st ? ` s="${cell.st}"` : "";
  if (cell.f !== undefined) return `<c r="${ref}"${st}><f>${esc(cell.f)}</f></c>`;
  if (cell.n !== undefined) return `<c r="${ref}"${st}><v>${num(cell.n)}</v></c>`;
  if (cell.s !== undefined)
    return `<c r="${ref}"${st} t="inlineStr"><is><t xml:space="preserve">${esc(cell.s)}</t></is></c>`;
  return `<c r="${ref}"${st}/>`;
}

export function sheetXml(sh: SheetSpec): string {
  const out: string[] = [HEAD, `<worksheet xmlns="${NS}">`];
  if (sh.cols?.length) {
    out.push("<cols>");
    for (const c of sh.cols)
      out.push(`<col min="${c.min}" max="${c.max}" width="${c.width}" customWidth="1"/>`);
    out.push("</cols>");
  }
  out.push("<sheetData>");
  const rows = Array.from(sh.cells.keys()).sort((a, b) => a - b);
  for (const r of rows) {
    const line = sh.cells.get(r)!;
    const cols = Array.from(line.keys()).sort((a, b) => a - b);
    out.push(`<row r="${r}">`);
    for (const c of cols) out.push(cellXml(r, c, line.get(c)!));
    out.push("</row>");
  }
  out.push("</sheetData>");
  // Attention heat maps. Priority must be unique per sheet.
  let prio = 1;
  for (const sqref of sh.colorScales ?? []) {
    out.push(
      `<conditionalFormatting sqref="${sqref}">` +
        `<cfRule type="colorScale" priority="${prio++}">` +
        `<colorScale><cfvo type="num" val="0"/><cfvo type="num" val="1"/>` +
        `<color rgb="FFFFFFFF"/><color rgb="FF1B2A3A"/></colorScale>` +
        `</cfRule></conditionalFormatting>`
    );
  }
  out.push("</worksheet>");
  return out.join("");
}

function workbookXml(wb: WorkbookSpec): string {
  const sheets = wb.sheets
    .map((s, i) => `<sheet name="${esc(s.name)}" sheetId="${i + 1}" r:id="rId${i + 1}"/>`)
    .join("");
  const names = (wb.definedNames ?? [])
    .map((d) => `<definedName name="${esc(d.name)}">${esc(d.ref)}</definedName>`)
    .join("");
  return (
    HEAD +
    `<workbook xmlns="${NS}" xmlns:r="${NS_R}">` +
    `<sheets>${sheets}</sheets>` +
    (names ? `<definedNames>${names}</definedNames>` : "") +
    // Excel writes no cached results here, so force a full recalculation on open.
    `<calcPr calcId="0" fullCalcOnLoad="1"/>` +
    `</workbook>`
  );
}

function stylesXml(): string {
  return (
    HEAD +
    `<styleSheet xmlns="${NS}">` +
    `<fonts count="5">` +
    `<font><sz val="11"/><name val="Calibri"/></font>` +
    `<font><b/><sz val="11"/><name val="Calibri"/></font>` +
    `<font><sz val="10"/><name val="Consolas"/></font>` +
    `<font><b/><sz val="11"/><name val="Consolas"/></font>` +
    `<font><b/><sz val="11"/><color rgb="FF1D3652"/><name val="Calibri"/></font>` +
    `</fonts>` +
    // Fargane er dei same som i appen: tusj-gult for det du kan endra, papir for
    // boksane i flytskjemaet, lys blå for eit svar.
    `<fills count="5"><fill><patternFill patternType="none"/></fill>` +
    `<fill><patternFill patternType="gray125"/></fill>` +
    `<fill><patternFill patternType="solid"><fgColor rgb="FFFFE68C"/><bgColor indexed="64"/></patternFill></fill>` +
    `<fill><patternFill patternType="solid"><fgColor rgb="FFF4F1E8"/><bgColor indexed="64"/></patternFill></fill>` +
    `<fill><patternFill patternType="solid"><fgColor rgb="FFE4ECF4"/><bgColor indexed="64"/></patternFill></fill>` +
    `</fills>` +
    `<borders count="3"><border/>` +
    `<border><left style="thin"><color rgb="FF1D3652"/></left>` +
    `<right style="thin"><color rgb="FF1D3652"/></right>` +
    `<top style="thin"><color rgb="FF1D3652"/></top>` +
    `<bottom style="thin"><color rgb="FF1D3652"/></bottom><diagonal/></border>` +
    `<border><left style="medium"><color rgb="FF1D3652"/></left>` +
    `<right style="medium"><color rgb="FF1D3652"/></right>` +
    `<top style="medium"><color rgb="FF1D3652"/></top>` +
    `<bottom style="medium"><color rgb="FF1D3652"/></bottom><diagonal/></border>` +
    `</borders>` +
    `<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>` +
    `<cellXfs count="9">` +
    `<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>` +
    `<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>` +
    `<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">` +
    `<alignment vertical="top" wrapText="1"/></xf>` +
    `<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>` +
    `<xf numFmtId="0" fontId="3" fillId="2" borderId="2" xfId="0" ` +
    `applyFont="1" applyFill="1" applyBorder="1"/>` +
    `<xf numFmtId="0" fontId="2" fillId="4" borderId="1" xfId="0" ` +
    `applyFont="1" applyFill="1" applyBorder="1"/>` +
    `<xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" ` +
    `applyFill="1" applyBorder="1" applyAlignment="1">` +
    `<alignment vertical="top" wrapText="1"/></xf>` +
    `<xf numFmtId="0" fontId="4" fillId="3" borderId="1" xfId="0" ` +
    `applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">` +
    `<alignment vertical="top" wrapText="1"/></xf>` +
    `<xf numFmtId="0" fontId="2" fillId="0" borderId="1" xfId="0" ` +
    `applyFont="1" applyBorder="1" applyAlignment="1">` +
    `<alignment vertical="top" wrapText="1"/></xf>` +
    `</cellXfs>` +
    `<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>` +
    `</styleSheet>`
  );
}

export interface Part {
  path: string;
  text: string;
}

/** Every OPC part Excel needs, as UTF-8 text. */
export function buildXlsxParts(wb: WorkbookSpec): Part[] {
  const n = wb.sheets.length;
  const overrides = wb.sheets
    .map(
      (_, i) =>
        `<Override PartName="/xl/worksheets/sheet${i + 1}.xml" ` +
        `ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`
    )
    .join("");

  const parts: Part[] = [
    {
      path: "[Content_Types].xml",
      text:
        HEAD +
        `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">` +
        `<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>` +
        `<Default Extension="xml" ContentType="application/xml"/>` +
        `<Override PartName="/xl/workbook.xml" ` +
        `ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>` +
        `<Override PartName="/xl/styles.xml" ` +
        `ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>` +
        overrides +
        `</Types>`,
    },
    {
      path: "_rels/.rels",
      text:
        HEAD +
        `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
        `<Relationship Id="rId1" ` +
        `Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" ` +
        `Target="xl/workbook.xml"/></Relationships>`,
    },
    { path: "xl/workbook.xml", text: workbookXml(wb) },
    {
      path: "xl/_rels/workbook.xml.rels",
      text:
        HEAD +
        `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
        wb.sheets
          .map(
            (_, i) =>
              `<Relationship Id="rId${i + 1}" Type="${NS_R}/worksheet" ` +
              `Target="worksheets/sheet${i + 1}.xml"/>`
          )
          .join("") +
        `<Relationship Id="rId${n + 1}" Type="${NS_R}/styles" Target="styles.xml"/>` +
        `</Relationships>`,
    },
    { path: "xl/styles.xml", text: stylesXml() },
  ];

  wb.sheets.forEach((sh, i) =>
    parts.push({ path: `xl/worksheets/sheet${i + 1}.xml`, text: sheetXml(sh) })
  );
  return parts;
}
