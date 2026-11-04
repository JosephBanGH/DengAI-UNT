import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, tokens } from "../api/client";
import type { Me } from "../api/types";

interface AuthState {
  user: Me | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<{ mustChangePassword: boolean }>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<Me | null>;
  can: (permission: string) => boolean;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

const ALL = "*";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    if (!tokens.access) {
      setUser(null);
      setLoading(false);
      return null;
    }
    try {
      const me = await api.get<Me>("/auth/me");
      setUser(me);
      tokens.setUser({ email: me.email, full_name: me.full_name, permissions: me.permissions });
      return me;
    } catch {
      tokens.clear();
      setUser(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshUser();
  }, [refreshUser]);

  const login = useCallback(async (email: string, password: string) => {
    const result = await api.post<{
      access_token: string;
      refresh_token: string;
      must_change_password: boolean;
    }>("/auth/login", { email, password });
    tokens.set(result.access_token, result.refresh_token);
    await refreshUser();
    return { mustChangePassword: result.must_change_password };
  }, [refreshUser]);

  const logout = useCallback(async () => {
    try {
      await api.post("/auth/logout");
    } catch {
      /* la sesión puede haber expirado */
    }
    tokens.clear();
    setUser(null);
  }, []);

  const can = useCallback(
    (permission: string) => {
      if (!user) return false;
      const granted = user.is_superuser ? [ALL] : user.permissions;
      return granted.includes(ALL) || granted.includes(permission);
    },
    [user],
  );

  const value = useMemo<AuthState>(
    () => ({ user, loading, login, logout, refreshUser, can }),
    [user, loading, login, logout, refreshUser, can],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth debe usarse dentro de AuthProvider");
  return context;
}
