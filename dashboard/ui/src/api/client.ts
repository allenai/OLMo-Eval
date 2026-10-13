import type { ErrorResponse } from "@contract/api-types";
import { apiQuery } from "@/lib/url";

/** Non-2xx API response with the server's error code, message and request ID. */
export class ApiError extends Error {
  status: number;
  code: string;
  requestId: string | null;
  details: unknown;

  constructor(status: number, code: string, message: string, requestId: string | null, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

const BASE = "/api";

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      signal,
      headers: {
        Accept: "application/json",
        // Required by the API on writes (CSRF guard). This exact value also makes IAP
        // answer an expired session with 401 instead of a redirect to Google sign-in.
        "X-Requested-With": "XMLHttpRequest",
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, "network_error", "Could not reach the API.", null);
  }
  const requestId = response.headers.get("X-Request-Id");
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  let data: unknown = null;
  let parsed = !text;
  if (text) {
    try {
      data = JSON.parse(text);
      parsed = true;
    } catch {
      data = null;
    }
  }
  if (response.ok && !parsed) {
    // Usually a sign-in or proxy error page served with status 200.
    throw new ApiError(
      response.status,
      "invalid_response",
      "The API returned a page instead of data. Your session may have expired; reload the page.",
      requestId,
    );
  }
  if (!response.ok) {
    const err = (data as ErrorResponse | null)?.error;
    throw new ApiError(
      response.status,
      err?.code ?? `http_${response.status}`,
      err?.message ?? (response.statusText || "Request failed"),
      err?.request_id ?? requestId,
      err?.details,
    );
  }
  return data as T;
}

export function apiGet<T>(path: string, params?: Record<string, unknown>, signal?: AbortSignal): Promise<T> {
  return request<T>("GET", `${path}${params ? apiQuery(params) : ""}`, undefined, signal);
}

export function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>("POST", path, body, signal);
}

export function apiPatch<T>(path: string, body: unknown): Promise<T> {
  return request<T>("PATCH", path, body);
}

export function apiDelete(path: string): Promise<void> {
  return request<void>("DELETE", path);
}
