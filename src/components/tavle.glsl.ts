// Shaderkjelde som inline strengar: bygget blir éi sjølvstendig HTML-fil,
// så ingenting kan hentast frå ein URL.

export const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  // DOM-teksturar er topp-ned, difor speglar vi y
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

export const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_board;   // levande DOM frå tavla
uniform sampler2D u_smudge;  // uklarleik per teikn, same geometri som tavla
uniform vec2 u_texel;        // 1.0 / oppløysing
uniform float u_time;        // sekund; frose ved prefers-reduced-motion
out vec4 outColor;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

void main() {
  float s = texture(u_smudge, v_uv).r;
  if (s <= 0.004) {
    outColor = texture(u_board, v_uv);
    return;
  }

  // Retninga kjem frå eit støyfelt, ikkje frå ein fast akse: krit smiter
  // ujamnt, slik ekte krit gjer.
  float a = noise(v_uv * 90.0 + u_time * 0.15) * 6.2831853;
  vec2 dir = vec2(cos(a), sin(a));

  // Fleire prøver langs retninga => kritstøvet blør inn i nabobokstaven.
  // Dette er heile grunnen til at tier 2 finst; CSS-uskarpleik er innestengd
  // i sin eigen boks per teikn og kan ikkje gjere dette.
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int k = 0; k < 6; k++) {
    float t = float(k) / 5.0;
    vec2 off = dir * s * t * u_texel * 7.0;
    float w = 1.0 - t * 0.7;
    acc += texture(u_board, v_uv + off) * w;
    wsum += w;
  }

  vec4 col = acc / wsum;
  col.a *= mix(1.0, 0.55, s);
  outColor = col;
}`;
