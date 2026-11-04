import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

type ToastKind = "info" | "success" | "error" | "warn";

interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastState {
  push: (message: string, kind?: ToastKind) => void;
}

const ToastContext = createContext<ToastState | undefined>(undefined);

let counter = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);

  const push = useCallback((message: string, kind: ToastKind = "info") => {
    const id = ++counter;
    setItems((current) => [...current, { id, kind, message }]);
    window.setTimeout(() => {
      setItems((current) => current.filter((item) => item.id !== id));
    }, 5200);
  }, []);

  const value = useMemo<ToastState>(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-stack" role="status" aria-live="polite">
        {items.map((item) => (
          <div key={item.id} className={`toast ${item.kind}`}>
            {item.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastState {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast debe usarse dentro de ToastProvider");
  return context;
}

export function describeError(error: unknown): string {
  if (error && typeof error === "object" && "message" in error) {
    const detail = (error as { details?: Record<string, unknown> }).details;
    const base = String((error as { message: string }).message);
    if (detail && Array.isArray(detail.missing)) return `${base} Falta: ${(detail.missing as string[]).join(", ")}`;
    if (detail && Array.isArray(detail.errors)) {
      const first = detail.errors[0] as { loc?: string[]; msg?: string };
      if (first?.msg) return `${base} (${(first.loc || []).slice(1).join(".")}: ${first.msg})`;
    }
    return base;
  }
  return "Ocurrió un error inesperado.";
}
