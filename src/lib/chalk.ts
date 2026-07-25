// Krittmatematikken. Alt som kan justerast bur her, slik at det kan
// finjusterast og testast i Node utan ein nettlesar – same mønster som
// SHARPEN i Skruer.tsx.
//
// Biletspråket er felles for begge tavlene: krit som ikkje har sett seg.
// Usikker modell => uklare, bleike bokstavar. Sikker modell => skarpt krit.

// Golvet er målt i praksis: preset «liten» på det norske korpuset legg seg
// rundt 1,25 i tap etter 3500 steg. Under golvet er biletet heilt skarpt.
const LOSS_FLOOR = 1.25;

// Maksimal uskarpleik i piksler. Over ~1,8px blir monospace-teikn uleselege
// heilt, og då er det ikkje lenger ei måling, berre grøt.
const MAX_BLUR_PX = 1.6;

// Kor bleikt det svakaste kritet blir. Under ~0,4 forsvinn teksten på tavla.
const MIN_OPACITY = 0.45;

// Tap => skarpleik. Nullpunktet er rein gjetting: eit tap på ln(V) tyder at
// modellen er like sikker på alle teikn i ordforrådet, altså at han ikkje
// veit noko som helst. Det er den ærlege botnen å måle frå.
export function lossToFocus(loss: number, vocab: number): number {
  // Eit øydelagt vokabular (NaN/uendeleg) gjev ikkje noko truverdig gjettetak
  // å måle frå. Då er det tryggaste å teikne tavla heilt uskarp, same
  // konvensjon som confToSmudge sitt garbage-in-fallback.
  if (!Number.isFinite(vocab)) return 0;
  const ceil = Math.log(Math.max(2, vocab));
  if (!Number.isFinite(loss) || loss <= 0) return 1;
  // Golvet kan aldri liggje meir enn halvvegs ned frå gjettetaket. Då er
  // span garantert positivt (span > ceil/2 > 0 sidan vocab >= 2), og
  // ln(V) landar difor alltid nøyaktig på 0 – også for eit lite vokabular
  // der det faste LOSS_FLOOR elles ville ha lege over eller på taket.
  const floor = Math.min(LOSS_FLOOR, ceil * 0.5);
  const span = ceil - floor;
  const t = (ceil - loss) / span;
  return Math.min(1, Math.max(0, t));
}

// Sikkerheit => uklarleik. Kvadratrota gjer at skilnaden mellom «ganske
// sikker» og «heilt sikker» ikkje får dominere biletet; det interessante
// skjer i den låge enden.
export function confToSmudge(conf: number): number {
  if (!Number.isFinite(conf)) return 1;
  const p = Math.min(1, Math.max(0, conf));
  return 1 - Math.sqrt(p);
}

export function meanConf(conf: Float32Array): number {
  if (conf.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < conf.length; i++) sum += conf[i];
  return sum / conf.length;
}

export function blurPx(smudge: number): number {
  const s = Math.min(1, Math.max(0, smudge));
  return s * MAX_BLUR_PX;
}

export function chalkOpacity(smudge: number): number {
  const s = Math.min(1, Math.max(0, smudge));
  return 1 - s * (1 - MIN_OPACITY);
}

// Tier 2 krev WebGL2 *og* den eksperimentelle html-in-canvas-utvidinga.
// Utan begge fell vi til tier 1, som er reint CSS og verkar overalt.
export function supportsElementTexture(): boolean {
  const proto = (globalThis as { WebGL2RenderingContext?: { prototype: unknown } })
    .WebGL2RenderingContext?.prototype as Record<string, unknown> | undefined;
  return typeof proto?.texElementImage2D === "function";
}

// ?tier=1 eller ?tier=2 tvingar eit nivå. Brukt til å måle kva tier 2
// faktisk kostar treninga – sjå målesteget i oppgåve 5.
export function forcedTier(): 1 | 2 | null {
  if (typeof location === "undefined") return null;
  const v = new URLSearchParams(location.search).get("tier");
  if (v === "1") return 1;
  if (v === "2") return 2;
  return null;
}
