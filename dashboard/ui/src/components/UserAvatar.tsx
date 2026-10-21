import Avatar from "boring-avatars";

/**
 * Ai2 palette for generated avatars: dark teal, light teal and pink (the light-theme --teal,
 * dark-theme --teal and --accent tokens). Fixed colors, so a person's avatar looks the same
 * in both themes. Every pair contrasts: near-white read as a hole against the page, and a
 * second dark teal made dark-on-dark faces.
 */
export const AVATAR_COLORS = ["#0a3235", "#7fc4bf", "#f0529c"];

/** A generated avatar that is stable for a given person. Seeded by email when known. */
export function UserAvatar({ seed, size = 28 }: { seed: string | undefined; size?: number }) {
  if (!seed) {
    return <span aria-hidden style={{ display: "block", width: size, height: size, borderRadius: "50%", background: "var(--row)" }} />;
  }
  return <Avatar name={seed.toLowerCase()} variant="beam" colors={AVATAR_COLORS} size={size} aria-hidden />;
}
