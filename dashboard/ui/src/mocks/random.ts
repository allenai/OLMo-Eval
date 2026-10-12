/** Deterministic randomness for the mock dataset. */

export function hashString(text: string, seed = 0): number {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return ((h2 >>> 0) ^ (h1 >>> 0)) >>> 0;
}

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function rngFor(...parts: (string | number)[]): Rng {
  return mulberry32(hashString(parts.join("|")));
}

export function normal(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function hexId(length: number, ...parts: (string | number)[]): string {
  const rng = rngFor("hex", ...parts);
  let out = "";
  for (let i = 0; i < length; i++) out += Math.floor(rng() * 16).toString(16);
  return out;
}

const CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/** 26-char Crockford base32, the shape of Beaker IDs. */
export function beakerId(...parts: (string | number)[]): string {
  const rng = rngFor("beaker", ...parts);
  let out = "01";
  for (let i = 0; i < 24; i++) out += CROCKFORD[Math.floor(rng() * 32)];
  return out;
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length) % items.length];
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
