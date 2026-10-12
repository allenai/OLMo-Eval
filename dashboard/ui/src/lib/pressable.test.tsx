import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Chip } from "@/components/primitives";
import { pressable } from "./pressable";

describe("pressable", () => {
  it("is focusable and activates on click, Enter and Space", () => {
    const onActivate = vi.fn();
    const { getByRole } = render(<div {...pressable(onActivate)}>row</div>);
    const row = getByRole("button");
    expect(row.tabIndex).toBe(0);
    fireEvent.click(row);
    fireEvent.keyDown(row, { key: "Enter" });
    fireEvent.keyDown(row, { key: " " });
    fireEvent.keyDown(row, { key: "a" });
    expect(onActivate).toHaveBeenCalledTimes(3);
  });

  it("leaves keys on nested controls alone", () => {
    const onActivate = vi.fn();
    const { getByText } = render(
      <div {...pressable(onActivate, "option")} aria-selected={false}>
        <button type="button">inner</button>
      </div>,
    );
    fireEvent.keyDown(getByText("inner"), { key: "Enter" });
    expect(onActivate).not.toHaveBeenCalled();
  });

  it("makes clickable chips reachable from the keyboard", () => {
    const onClick = vi.fn();
    const { getByRole } = render(<Chip value="tag" onClick={onClick} active={false} />);
    fireEvent.keyDown(getByRole("button"), { key: "Enter" });
    expect(onClick).toHaveBeenCalledOnce();
  });
});
