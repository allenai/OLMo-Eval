import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { type Column, DataTable, nextSort, parseSortList, sortListParam, sortRows } from "./DataTable";

interface Row {
  id: string;
  name: string;
  score: number | null;
}

const rows: Row[] = [
  { id: "a", name: "alpha", score: 0.5 },
  { id: "b", name: "beta", score: 0.9 },
  { id: "c", name: "gamma", score: null },
];

const columns: Column<Row>[] = [
  { id: "name", header: "Name", sortValue: (r) => r.name, cell: (r) => r.name, csv: (r) => r.name },
  { id: "score", header: "Score", sortValue: (r) => r.score, cell: (r) => String(r.score ?? "—"), csv: (r) => r.score },
];

describe("sorting helpers", () => {
  it("cycles desc, asc, none and supports additive sorts", () => {
    expect(nextSort([], "a", false)).toEqual([{ key: "a", desc: true }]);
    expect(nextSort([{ key: "a", desc: true }], "a", false)).toEqual([{ key: "a", desc: false }]);
    expect(nextSort([{ key: "a", desc: false }], "a", false)).toEqual([]);
    expect(nextSort([{ key: "a", desc: true }], "b", true)).toEqual([
      { key: "a", desc: true },
      { key: "b", desc: true },
    ]);
  });

  it("round-trips the URL sort param", () => {
    expect(sortListParam(parseSortList("-score,name"))).toBe("-score,name");
    expect(sortListParam([])).toBeUndefined();
  });

  it("sorts with nulls last in both directions", () => {
    expect(sortRows(rows, [{ key: "score", desc: true }], columns).map((r) => r.id)).toEqual(["b", "a", "c"]);
    expect(sortRows(rows, [{ key: "score", desc: false }], columns).map((r) => r.id)).toEqual(["a", "b", "c"]);
  });
});

describe("DataTable", () => {
  it("renders rows, sorts on header click and reports selection", () => {
    const onSelectionChange = vi.fn();
    render(
      <DataTable tableId="test" columns={columns} rows={rows} getRowId={(r) => r.id} selected={new Set()} onSelectionChange={onSelectionChange} maxHeight={400} rowHeight={30} />,
    );
    expect(screen.getByText("3 rows")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("columnheader", { name: /Score/ }));
    expect(screen.getByRole("columnheader", { name: /Score/ })).toHaveAttribute("aria-sort", "descending");
    fireEvent.click(screen.getByLabelText("Select all rows"));
    expect(onSelectionChange).toHaveBeenCalledWith(new Set(["a", "b", "c"]));
  });

  it("opens rows on click and shows an empty state", () => {
    const onRowOpen = vi.fn();
    const { rerender } = render(<DataTable tableId="t2" columns={columns} rows={rows} getRowId={(r) => r.id} onRowOpen={onRowOpen} maxHeight={400} />);
    const grid = screen.getByRole("grid");
    const firstRow = within(grid).getAllByRole("row")[1];
    fireEvent.click(firstRow);
    expect(onRowOpen).toHaveBeenCalled();
    rerender(<DataTable tableId="t2" columns={columns} rows={[]} getRowId={(r) => r.id} empty={<span>Nothing here</span>} />);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("puts row actions in their own trailing column", () => {
    // Regression: the actions used to be placed across every column, which pushed the row's
    // cells into a second grid row on hover and covered the links in the last column.
    render(
      <DataTable
        tableId="t3"
        columns={columns}
        rows={rows}
        getRowId={(r) => r.id}
        maxHeight={400}
        rowActions={(r) => <button type="button">act {r.id}</button>}
      />,
    );
    const grid = screen.getByRole("grid");
    const [header, firstRow] = within(grid).getAllByRole("row");
    const template = firstRow.style.gridTemplateColumns;
    expect(template.split(" ").at(-1)).toBe("96px");
    expect(header.style.gridTemplateColumns).toBe(template);
    expect(header.children).toHaveLength(columns.length + 1);
    const cells = within(firstRow).getAllByRole("gridcell");
    expect(cells).toHaveLength(columns.length + 1);
    expect(cells.at(-1)).toHaveTextContent("act a");
    expect(cells.at(-1)?.getAttribute("style")).toBeNull();
  });
});
