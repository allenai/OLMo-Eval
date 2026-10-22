import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiGet } from "./client";

function respond(status: number, body: string, headers: Record<string, string> = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(body, { status, headers })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("apiGet", () => {
  it("returns parsed JSON", async () => {
    respond(200, JSON.stringify({ items: [1] }));
    await expect(apiGet("/runs")).resolves.toEqual({ items: [1] });
  });

  it("rejects a 200 response that is not JSON", async () => {
    respond(200, "<html>Sign in</html>", { "X-Request-Id": "req-1" });
    const err = await apiGet("/runs").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).code).toBe("invalid_response");
    expect((err as ApiError).requestId).toBe("req-1");
  });

  it("uses the server's error body on failures", async () => {
    respond(403, JSON.stringify({ error: { code: "forbidden", message: "no", request_id: "r" } }));
    const err = (await apiGet("/runs").catch((e: unknown) => e)) as ApiError;
    expect(err.status).toBe(403);
    expect(err.code).toBe("forbidden");
  });
});
