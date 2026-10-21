import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScoreValue } from "./cells";

describe("ScoreValue generation diagnostics", () => {
  it("shows saved generation counts beside a score", () => {
    render(<ScoreValue score={0.5} generationCounts={{ generations: 4, cap_hit: 2, empty: 1, unclosed_think: 1 }} />);
    expect(screen.getByText("50.0")).toBeInTheDocument();
    expect(screen.getByText("cap 2/4 · empty 1 · unclosed 1")).toBeInTheDocument();
  });

  it("reports unknown finish reasons without implying explicit thinking closed", () => {
    render(<ScoreValue score={null} generationCounts={{ generations: 3, cap_hit: 0, empty: 0, unclosed_think: 0, finish_reason_unknown: 3 }} />);
    expect(screen.getByText("cap 0/3 · empty 0 · unknown 3")).toBeInTheDocument();
    expect(screen.queryByText(/unclosed/)).not.toBeInTheDocument();
  });

  it("keeps legacy and loglikelihood scores free of unavailable diagnostics", () => {
    const { rerender } = render(<ScoreValue score={0.5} />);
    expect(screen.queryByText(/cap /)).not.toBeInTheDocument();
    rerender(<ScoreValue score={0.5} generationCounts={{ generations: 0 }} />);
    expect(screen.queryByText(/cap /)).not.toBeInTheDocument();
  });
});
