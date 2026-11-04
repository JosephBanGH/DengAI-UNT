import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { Icon, type IconName } from "./Icons";

interface NavEntry {
  to: string;
  label: string;
  icon: IconName;
  permission: string;
  group: string;
}

const NAV: NavEntry[] = [
  { to: "/", label: "Dashboard ejecutivo", icon: "dashboard", permission: "dashboard:read", group: "Vigilancia" },
  { to: "/mapa", label: "Mapas de riesgo", icon: "map", permission: "geo:read", group: "Vigilancia" },
  { to: "/pronosticos", label: "Pronósticos", icon: "forecast", permission: "predictions:read", group: "Vigilancia" },
  { to: "/alertas", label: "Alertas tempranas", icon: "alert", permission: "alerts:read", group: "Vigilancia" },
  { to: "/datos", label: "Datos y calidad", icon: "data", permission: "data:read", group: "Datos" },
  { to: "/conjuntos", label: "Conjuntos de datos", icon: "dataset", permission: "datasets:read", group: "Datos" },
  { to: "/laboratorio", label: "Laboratorio de IA", icon: "ml", permission: "ml:read", group: "Inteligencia artificial" },
  { to: "/reportes", label: "Reportes", icon: "report", permission: "reports:read", group: "Inteligencia artificial" },
  { to: "/asistente", label: "Asistente inteligente", icon: "assistant", permission: "assistant:ask", group: "Inteligencia artificial" },
  { to: "/workflows", label: "Automatización", icon: "workflow", permission: "workflows:read", group: "Operación" },
  { to: "/auditoria", label: "Bitácora y auditoría", icon: "audit", permission: "audit:read", group: "Administración" },
  { to: "/usuarios", label: "Usuarios", icon: "users", permission: "users:read", group: "Administración" },
  { to: "/roles", label: "Roles y permisos", icon: "roles", permission: "roles:read", group: "Administración" },
  { to: "/configuracion", label: "Configuración", icon: "settings", permission: "admin:config", group: "Administración" },
];

const PUBLIC_TITLES: Record<string, string> = {
  "/perfil": "Mi perfil",
};

export function Layout({ children }: { children?: ReactNode }) {
  const { user, can, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [synthetic, setSynthetic] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const visible = NAV.filter((entry) => can(entry.permission));
  const groups = Array.from(new Set(visible.map((entry) => entry.group)));
  const current = visible.find((entry) => entry.to === location.pathname);

  useEffect(() => {
    let cancelled = false;
    api
      .get<{ is_synthetic_data: boolean }>("/dashboard/overview")
      .then((data) => {
        if (!cancelled) setSynthetic(Boolean(data.is_synthetic_data));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const module = current?.to || location.pathname;
    api
      .post("/audit/module-event", { event_type: "MODULE_ENTER", module, action: "navigate" })
      .catch(() => undefined);
    const handler = () => {
      // fetch con keepalive envía el token en el encabezado (sendBeacon no permite cabeceras).
      const token = localStorage.getItem("dengai.access");
      if (!token) return;
      fetch("/api/v1/audit/module-event", {
        method: "POST",
        keepalive: true,
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ event_type: "MODULE_EXIT", module, action: "leave" }),
      }).catch(() => undefined);
    };
    window.addEventListener("beforeunload", handler);
    return () => {
      window.removeEventListener("beforeunload", handler);
      api
        .post("/audit/module-event", { event_type: "MODULE_EXIT", module, action: "route_change" })
        .catch(() => undefined);
    };
  }, [current?.to, location.pathname]);

  const title = current?.label || PUBLIC_TITLES[location.pathname] || "SP-01 Dengue Intelligence";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <div className="brand-logo">D1</div>
            <div>
              <div className="brand-name">Dengue Intelligence</div>
              <div className="brand-sub">SP-01 · UNT</div>
            </div>
          </div>
        </div>
        <nav className="nav">
          {groups.map((group) => (
            <div className="nav-group" key={group}>
              <div className="nav-group-title">{group}</div>
              {visible
                .filter((entry) => entry.group === group)
                .map((entry) => (
                  <NavLink
                    key={entry.to}
                    to={entry.to}
                    end={entry.to === "/"}
                    className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
                  >
                    <Icon name={entry.icon} className="nav-icon" />
                    {entry.label}
                  </NavLink>
                ))}
            </div>
          ))}
        </nav>
        <div style={{ padding: "12px 16px", borderTop: "1px solid rgba(255,255,255,0.08)", fontSize: 11, color: "#7fa3b1" }}>
          v1.0.0 · apoyo a la decisión epidemiológica
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div>
            <div className="topbar-title">{title}</div>
            <div className="topbar-sub">
              {user ? `${user.full_name} · ${user.roles.map((role) => role.name).join(", ")}` : "Sesión no iniciada"}
            </div>
          </div>
          <div className="topbar-spacer" />
          <button className="btn btn-ghost btn-sm" onClick={() => setMenuOpen((value) => !value)}>
            <Icon name="users" /> {user?.email?.split("@")[0]}
          </button>
          {menuOpen ? (
            <div
              className="card"
              style={{ position: "absolute", right: 24, top: 54, width: 230, zIndex: 40, padding: 8 }}
              onMouseLeave={() => setMenuOpen(false)}
            >
              <button
                className="nav-item"
                style={{ color: "var(--text)" }}
                onClick={() => {
                  setMenuOpen(false);
                  navigate("/perfil");
                }}
              >
                <Icon name="settings" className="nav-icon" /> Mi perfil y sesiones
              </button>
              <button
                className="nav-item"
                style={{ color: "var(--danger)" }}
                onClick={async () => {
                  await logout();
                  navigate("/login");
                }}
              >
                <Icon name="logout" className="nav-icon" /> Cerrar sesión
              </button>
            </div>
          ) : null}
        </header>

        <main className="content">
          {synthetic ? (
            <div className="synthetic-banner">
              <Icon name="info" />
              <span>
                <strong>Datos sintéticos de demostración.</strong> Las cifras provienen de un conjunto generado con semilla
                fija (La Libertad, 14 distritos) y no representan vigilancia epidemiológica real.
              </span>
            </div>
          ) : null}
          {children ?? <Outlet />}
        </main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div>
        <div className="page-title">{title}</div>
        {description ? <div className="page-desc">{description}</div> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  );
}
