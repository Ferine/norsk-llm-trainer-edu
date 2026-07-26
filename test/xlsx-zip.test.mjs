// Exercises the zip writer and the download path. The ZIP is re-read with
// node:zlib, independently of the code that wrote it, so a bad CRC, a wrong
// header length or a broken deflate stream shows up here rather than in Excel.

import assert from "node:assert/strict";
import { inflateRawSync } from "node:zlib";
import { zip, workbookToBlob, downloadWorkbook } from "./dist/xlsx-zip.js";
import { newSheet, put, buildXlsxParts } from "./dist/xlsx.js";

// ---- a hand-rolled ZIP reader, so the test does not trust the writer --------
function readZip(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // Locate the end-of-central-directory record.
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--)
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  assert.ok(eocd >= 0, "no end-of-central-directory record");
  const count = dv.getUint16(eocd + 10, true);
  const dirSize = dv.getUint32(eocd + 12, true);
  const dirOff = dv.getUint32(eocd + 16, true);
  assert.equal(dirOff + dirSize, eocd, "central directory should end at the EOCD");

  const out = new Map();
  let p = dirOff;
  for (let i = 0; i < count; i++) {
    assert.equal(dv.getUint32(p, true), 0x02014b50, `bad central header ${i}`);
    const method = dv.getUint16(p + 10, true);
    const crc = dv.getUint32(p + 16, true);
    const packed = dv.getUint32(p + 20, true);
    const raw = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const cmtLen = dv.getUint16(p + 32, true);
    const localOff = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(buf.subarray(p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + cmtLen;

    // Follow the offset into the local header and pull the bytes out.
    assert.equal(dv.getUint32(localOff, true), 0x04034b50, `bad local header for ${name}`);
    const lNameLen = dv.getUint16(localOff + 26, true);
    const lExtraLen = dv.getUint16(localOff + 28, true);
    const dataAt = localOff + 30 + lNameLen + lExtraLen;
    const packedBytes = buf.subarray(dataAt, dataAt + packed);
    const bytes = method === 8 ? inflateRawSync(packedBytes) : Buffer.from(packedBytes);
    assert.equal(bytes.length, raw, `size mismatch for ${name}`);
    assert.equal(crc32(bytes), crc, `CRC mismatch for ${name}`);
    out.set(name, new TextDecoder().decode(bytes));
  }
  assert.equal(p, eocd, "central directory should be fully consumed");
  return out;
}

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c ^= bytes[i];
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return (c ^ 0xffffffff) >>> 0;
}

// ---- round-trip a workbook -------------------------------------------------
const sh = newSheet("Ark");
put(sh, 1, 1, { s: 'tekst med & < > og "hermeteikn"' });
put(sh, 1, 2, { n: -0.10000000149011612 });
put(sh, 2, 1, { f: 'IF(Ark!$A$1<>"",1&"x",0)' });
put(sh, 2, 2, { n: 0 });
const wb = { sheets: [sh], definedNames: [{ name: "Ting", ref: "Ark!$A$1" }] };

const blob = await workbookToBlob(wb);
assert.equal(
  blob.type,
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
);
const buf = Buffer.from(await blob.arrayBuffer());
const entries = readZip(buf);

const expected = buildXlsxParts(wb);
assert.equal(entries.size, expected.length, "entry count");
for (const part of expected) {
  assert.ok(entries.has(part.path), `missing ${part.path}`);
  assert.equal(entries.get(part.path), part.text, `content of ${part.path}`);
}
// The escaping has to survive the round-trip literally.
assert.ok(entries.get("xl/worksheets/sheet1.xml").includes("&amp; &lt; &gt;"));
assert.ok(entries.get("xl/worksheets/sheet1.xml").includes("&lt;&gt;&quot;&quot;"));
console.log(`  zip: ${entries.size} entries round-tripped, CRCs and sizes verified`);

// A stored-only fallback must produce an equally readable file.
const noCompress = globalThis.CompressionStream;
try {
  delete globalThis.CompressionStream;
  const stored = Buffer.from(await (await zip(expected)).arrayBuffer());
  const back = readZip(stored);
  assert.equal(back.size, expected.length);
  for (const part of expected) assert.equal(back.get(part.path), part.text);
  console.log(`  zip: stored fallback also readable (${stored.length} bytes, no deflate)`);
} finally {
  if (noCompress) globalThis.CompressionStream = noCompress;
}

// ---- the download path, against a stub DOM --------------------------------
const clicks = [];
const revoked = [];
const created = [];
globalThis.URL.createObjectURL = (b) => {
  created.push(b);
  return "blob:stub/1";
};
globalThis.URL.revokeObjectURL = (u) => revoked.push(u);
const appended = [];
globalThis.document = {
  createElement: () => ({
    click() {
      clicks.push({ href: this.href, download: this.download, rel: this.rel });
    },
    remove() {
      appended.pop();
    },
  }),
  body: { appendChild: (el) => appended.push(el) },
};

// downloadWorkbook parks the object-URL revoke on a long timer. Capture that one
// and run it by hand, so the test asserts the cleanup instead of waiting for it.
const realSetTimeout = globalThis.setTimeout;
const parked = [];
globalThis.setTimeout = (fn, ms, ...rest) =>
  ms >= 1000 ? (parked.push(fn), 0) : realSetTimeout(fn, ms, ...rest);
try {
  await downloadWorkbook(wb, "modell-liten-steg600.xlsx");
} finally {
  globalThis.setTimeout = realSetTimeout;
}

assert.equal(parked.length, 1, "should park exactly one revoke");
assert.deepEqual(revoked, [], "must not revoke before the download starts");
parked[0]();
assert.deepEqual(revoked, ["blob:stub/1"], "should revoke the object URL");

assert.equal(clicks.length, 1, "should click exactly one anchor");
assert.equal(clicks[0].download, "modell-liten-steg600.xlsx");
assert.equal(clicks[0].href, "blob:stub/1");
assert.equal(clicks[0].rel, "noopener");
assert.equal(created.length, 1);
assert.equal(appended.length, 0, "the anchor should be removed again");
console.log("  download: anchor created, clicked with the right filename, URL revoked");

console.log("xlsx-zip: PASS");
