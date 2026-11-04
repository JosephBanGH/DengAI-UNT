import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import type { ReactElement } from "react";
import { Layout } from "./components/Layout";
import { Loading } from "./components/ui";
import { useAuth } from "./context/AuthContext";
import { AssistantPage } from "./pages/Assistant";
import { AlertsPage } from "./pages/Alerts";
import { AuditPage } from "./pages/Audit";
import { DashboardPage } from "./pages/Dashboard";
import { DataPage } from "./pages/Data";
import { DatasetsPage } from "./pages/Datasets";
import { ExperimentDetailPage } from "./pages/ExperimentDetail";
import { ForecastsPage } from "./pages/Forecasts";
import { LoginPage } from "./pages/Login";
import { MlLabPage } from "./pages/MlLab";
import { ProfilePage } from "./pages/Profile";
import { ReportsPage } from "./pages/Reports";
import { RiskMapPage } from "./pages/RiskMap";
import { RolesPage } from "./pages/Roles";
import { SettingsPage } from "./pages/Settings";
import { UsersPage } from "./pages/Users";
import { WorkflowsPage } from "./pages/Workflows";

function RequireAuth({ children, permission }: { children?: ReactElement; permission?: string }) {
  const { user, loading, can } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
        <Loading label="Verificando sesión…" />
      </div>
    );
  }
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  if (permission && !can(permission)) {
    return (
      <Layout>
        <div className="card card-pad">
          <h2 style={{ marginBottom: 6 }}>Acceso restringido</h2>
          <p className="muted">
            Su rol no cuenta con el permiso <span className="mono">{permission}</span> requerido para este módulo.
            Solicite acceso al administrador del sistema.
          </p>
        </div>
      </Layout>
    );
  }
  return children ?? <Layout />;
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route path="/" element={<DashboardPage />} />
        <Route path="/mapa" element={<RiskMapPage />} />
        <Route path="/pronosticos" element={<ForecastsPage />} />
        <Route path="/alertas" element={<AlertsPage />} />
        <Route path="/datos" element={<DataPage />} />
        <Route path="/conjuntos" element={<DatasetsPage />} />
        <Route path="/laboratorio" element={<MlLabPage />} />
        <Route path="/laboratorio/:id" element={<ExperimentDetailPage />} />
        <Route path="/reportes" element={<ReportsPage />} />
        <Route path="/asistente" element={<AssistantPage />} />
        <Route path="/workflows" element={<WorkflowsPage />} />
        <Route path="/auditoria" element={<AuditPage />} />
        <Route path="/usuarios" element={<UsersPage />} />
        <Route path="/roles" element={<RolesPage />} />
        <Route path="/configuracion" element={<SettingsPage />} />
        <Route path="/perfil" element={<ProfilePage />} />
        <Route
          path="*"
          element={
            <div className="card card-pad">
              <h2>Página no encontrada</h2>
              <p className="muted">La ruta solicitada no existe en la plataforma.</p>
            </div>
          }
        />
      </Route>
    </Routes>
  );
}
