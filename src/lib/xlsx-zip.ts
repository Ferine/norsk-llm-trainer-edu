// ============================================================================
// ZIP writer + download trigger. An .xlsx is just a ZIP of the XML parts from
// xlsx.ts, so this is all it takes: CRC-32, deflate via the platform's own
// CompressionStream, and the two ZIP headers. No dependencies.
// ============================================================================

import { buildXlsxParts, type WorkbookSpec } from "./xlsx.js";

let CRC_TABLE: Uint32Array | null = null;

function crcTable(): Uint32Array {
  if (CRC_TABLE) return CRC_TABLE;
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  CRC_TABLE = t;
  return t;
}

function crc32(bytes: Uint8Array): number {
  const t = crcTable();
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = t[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// Raw DEFLATE via the platform. Returns null when unavailable so the caller can
// fall back to a stored (uncompressed) entry — still a perfectly valid .xlsx.
async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array | null> {
  if (typeof CompressionStream === "undefined") return null;
  try {
    const cs = new CompressionStream("deflate-raw");
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(cs);
    const buf = await new Response(stream).arrayBuffer();
    return new Uint8Array(buf);
  } catch {
    return null;
  }
}

/** MS-DOS date/time pair used by the ZIP headers. */
function dosStamp(d: Date): { time: number; date: number } {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}

class ByteSink {
  private chunks: Uint8Array[] = [];
  length = 0;

  push(b: Uint8Array) {
    this.chunks.push(b);
    this.length += b.length;
  }
  u16(v: number) {
    const b = new Uint8Array(2);
    new DataView(b.buffer).setUint16(0, v & 0xffff, true);
    this.push(b);
  }
  u32(v: number) {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v >>> 0, true);
    this.push(b);
  }
  parts(): Uint8Array[] {
    return this.chunks;
  }
}

interface Entry {
  name: Uint8Array;
  crc: number;
  raw: number;
  packed: Uint8Array;
  method: number;
  offset: number;
}

export async function zip(files: { path: string; text: string }[]): Promise<Blob> {
  const enc = new TextEncoder();
  const { time, date } = dosStamp(new Date());
  const body = new ByteSink();
  const entries: Entry[] = [];

  for (const f of files) {
    const bytes = enc.encode(f.text);
    const name = enc.encode(f.path);
    const crc = crc32(bytes);
    const deflated = await deflateRaw(bytes);
    // Only take the compressed form if it actually helped.
    const useDeflate = deflated !== null && deflated.length < bytes.length;
    const packed = useDeflate ? deflated! : bytes;
    const method = useDeflate ? 8 : 0;
    const offset = body.length;

    body.u32(0x04034b50); // local file header
    body.u16(useDeflate ? 20 : 10); // version needed
    body.u16(0x0800); // flags: UTF-8 names
    body.u16(method);
    body.u16(time);
    body.u16(date);
    body.u32(crc);
    body.u32(packed.length);
    body.u32(bytes.length);
    body.u16(name.length);
    body.u16(0); // extra length
    body.push(name);
    body.push(packed);

    entries.push({ name, crc, raw: bytes.length, packed, method, offset });
  }

  const dir = new ByteSink();
  for (const e of entries) {
    dir.u32(0x02014b50); // central directory header
    dir.u16(0x031e); // made by: UNIX, spec 3.0
    dir.u16(e.method === 8 ? 20 : 10);
    dir.u16(0x0800);
    dir.u16(e.method);
    dir.u16(time);
    dir.u16(date);
    dir.u32(e.crc);
    dir.u32(e.packed.length);
    dir.u32(e.raw);
    dir.u16(e.name.length);
    dir.u16(0); // extra
    dir.u16(0); // comment
    dir.u16(0); // disk
    dir.u16(0); // internal attrs
    dir.u32(0); // external attrs
    dir.u32(e.offset);
    dir.push(e.name);
  }

  const end = new ByteSink();
  end.u32(0x06054b50); // end of central directory
  end.u16(0);
  end.u16(0);
  end.u16(entries.length);
  end.u16(entries.length);
  end.u32(dir.length);
  end.u32(body.length);
  end.u16(0); // comment length

  const blobParts: BlobPart[] = [
    ...body.parts(),
    ...dir.parts(),
    ...end.parts(),
  ] as BlobPart[];
  return new Blob(blobParts, {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}

export async function workbookToBlob(wb: WorkbookSpec): Promise<Blob> {
  return zip(buildXlsxParts(wb));
}

export async function downloadWorkbook(wb: WorkbookSpec, filename: string): Promise<void> {
  const blob = await workbookToBlob(wb);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the browser a tick to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
