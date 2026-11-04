const API_BASE = "/api/v1";

const ACCESS_KEY = "dengai.access";
const REFRESH_KEY = "dengai.refresh";
const USER_KEY = "dengai.user";

export interface ApiErrorShape {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export class ApiError extends Error {
  code: string;
  status: number;
  details: Record<string, unknown>;

  constructor(status: number, body: ApiErrorShape) {
    super(body.message || "Error de comunicación con el servidor");
    this.status = status;
    this.code = body.code || `HTTP_${status}`;
    this.details = body.details || {};
  }
}

export const tokens = {
  get access() {
    return localStorage.getItem(ACCESS_KEY);
  },
  get refresh() {
    return localStorage.getItem(REFRESH_KEY);
  },
  set(access: string, refresh?: string) {
    localStorage.setItem(ACCESS_KEY, access);
    if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
  },
  get userCache() {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as { email: string; full_name: string; permissions: string[] };
    } catch {
      return null;
    }
  },
  setUser(user: unknown) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
};

let refreshPromise: Promise<boolean> | null = null;

async function tryRefresh(): Promise<boolean> {
  const refreshToken = tokens.refresh;
  if (!refreshToken) return false;
  if (!refreshPromise) {
    refreshPromise = (async () => {
      try {
        const response = await fetch(`${API_BASE}/auth/refresh`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!response.ok) return false;
        const data = await response.json();
        tokens.set(data.access_token, data.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        refreshPromise = null;
      }
    })();
  }
  return refreshPromise;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined | null>;
  formData?: FormData;
  skipAuth?: boolean;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const url = new URL(`${window.location.origin}${API_BASE}${path}`);
  if (query) {
    Object.entries(query).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") return;
      url.searchParams.set(key, String(value));
    });
  }
  return url.pathname + url.search;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, query, formData, skipAuth } = options;
  const headers: Record<string, string> = {};
  if (!formData) headers["Content-Type"] = "application/json";
  const access = tokens.access;
  if (access && !skipAuth) headers.Authorization = `Bearer ${access}`;

  const execute = () =>
    fetch(buildUrl(path, query), {
      method,
      headers,
      body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
    });

  let response = await execute();

  if (response.status === 401 && !skipAuth) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      headers.Authorization = `Bearer ${tokens.access}`;
      response = await execute();
    } else {
      tokens.clear();
      if (!window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    }
  }

  if (!response.ok) {
    let payload: ApiErrorShape = { code: `HTTP_${response.status}`, message: response.statusText };
    try {
      const parsed = await response.json();
      if (parsed?.error) payload = parsed.error;
      else if (parsed?.detail) payload = typeof parsed.detail === "string" ? { code: `HTTP_${response.status}`, message: parsed.detail } : parsed.detail;
    } catch {
      /* respuesta sin cuerpo JSON */
    }
    throw new ApiError(response.status, payload);
  }

  if (response.status === 204) return undefined as T;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return (await response.json()) as T;
  return (await response.blob()) as unknown as T;
}

export const api = {
  get: <T>(path: string, query?: RequestOptions["query"]) => request<T>(path, { query }),
  post: <T>(path: string, body?: unknown) => request<T>(path, { method: "POST", body }),
  patch: <T>(path: string, body?: unknown) => request<T>(path, { method: "PATCH", body }),
  put: <T>(path: string, body?: unknown) => request<T>(path, { method: "PUT", body }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  upload: <T>(path: string, formData: FormData) => request<T>(path, { method: "POST", formData }),
};

export async function downloadFile(path: string, filename: string): Promise<void> {
  const blob = await request<Blob>(path);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
