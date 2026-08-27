export interface ContextWindowSlice {
  total: number;
  used: number;
  forgotten: number;
  forgottenPreview: string[];
  visible: string[];
}

// Del den faktiske tokenrekkja på same stad som genereringa gjer. Komponenten
// får ferdig dekoda teikn-token, så ukjende teikn som tokenizeren hoppar over,
// blir heller ikkje feilaktig viste som noko modellen kan sjå.
export function sliceContextWindow(
  tokens: readonly string[],
  capacity: number,
  previewCount = 8
): ContextWindowSlice {
  if (!Number.isInteger(capacity) || capacity < 1)
    throw new RangeError("context capacity must be a positive integer");
  if (!Number.isInteger(previewCount) || previewCount < 0)
    throw new RangeError("forgotten preview count must be a non-negative integer");

  const forgotten = Math.max(0, tokens.length - capacity);
  return {
    total: tokens.length,
    used: Math.min(tokens.length, capacity),
    forgotten,
    forgottenPreview: tokens.slice(Math.max(0, forgotten - previewCount), forgotten),
    visible: tokens.slice(forgotten),
  };
}
