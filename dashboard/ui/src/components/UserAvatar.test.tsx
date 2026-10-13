import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { UserAvatar } from "./UserAvatar";

// The SVG mask id comes from React's useId and differs per render, so compare without it.
const markup = (seed: string | undefined) =>
  render(<UserAvatar seed={seed} />).container.innerHTML.replace(/_r_\w+_/g, "ID");

describe("UserAvatar", () => {
  it("is stable per person and ignores email case", () => {
    expect(markup("chrisg@allenai.org")).toBe(markup("chrisg@allenai.org"));
    expect(markup("ChrisG@AllenAI.org")).toBe(markup("chrisg@allenai.org"));
  });

  it("differs between people", () => {
    expect(markup("chrisg@allenai.org")).not.toBe(markup("someone@allenai.org"));
  });

  it("renders a placeholder until the user is known", () => {
    const html = markup(undefined);
    expect(html).not.toContain("<svg");
    expect(html).toContain("border-radius: 50%");
  });
});
