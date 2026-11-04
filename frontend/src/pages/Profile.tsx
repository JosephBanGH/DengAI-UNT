import { useEffect, useState } from "react";
import { api } from "../api/client";
import { PageHeader } from "../components/Layout";
import { Card, CardBody, CardHeader, DataTable, EmptyState, ErrorAlert, Loading, formatDate } from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const toast = useToast();
  const [sessions, setSessions] = useState<any[]>([]);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSessions = () => {
    api
      .get<any[]>("/auth/sessions")
      .then((data) => {
        setSessions(data);
        setCurrent(data.find((item) => item.current)?.public_id || "");
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(loadSessions, []);

  const changePassword = async () => {
    try {
      await api.post("/auth/change-password", { current_password: "Admin#Dengue2025", new_password: next });
      toast.push("Contraseña actualizada.", "success");
      setNext("");
      await refreshUser();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const revoke = async (publicId: string) => {
    try {
      await api.del(`/auth/sessions/${publicId}`);
      toast.push("Sesión revocada.", "success");
      loadSessions();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  return (
    <>
      <PageHeader title="Mi perfil y seguridad" description="Datos de la cuenta, cambio de contraseña y control de sesiones activas." />

      {error ? <ErrorAlert message={error} /> : null}

      <div className="grid grid-main">
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <CardHeader title="Datos de la cuenta" subtitle="Información institucional del usuario" />
            <CardBody>
              <div className="grid grid-2">
                <div><div className="kpi-label">Nombre</div><strong>{user?.full_name}</strong></div>
                <div><div className="kpi-label">Correo</div><strong>{user?.email}</strong></div>
                <div><div className="kpi-label">Cargo</div><span>{user?.position || "—"}</span></div>
                <div><div className="kpi-label">Organización</div><span>{user?.organization || "—"}</span></div>
                <div><div className="kpi-label">Roles</div><span>{user?.roles.map((role) => role.name).join(", ")}</span></div>
                <div><div className="kpi-label">Último acceso</div><span>{formatDate(user?.last_login_at || null, true)}</span></div>
              </div>
              <p className="hint" style={{ marginTop: 10 }}>
                {user?.permissions.length} permisos efectivos en la plataforma.
              </p>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Sesiones activas" subtitle="Dispositivos y contextos de acceso" />
            <CardBody>
              {loading ? (
                <Loading />
              ) : sessions.length ? (
                <DataTable
                  rows={sessions}
                  rowKey={(row) => row.public_id}
                  columns={[
                    { key: "public_id", header: "Sesión", render: (row) => <span className="mono">{row.public_id}</span> },
                    { key: "created_at", header: "Inicio", render: (row) => formatDate(row.created_at, true) },
                    { key: "last_seen_at", header: "Última actividad", render: (row) => formatDate(row.last_seen_at, true) },
                    { key: "ip_address", header: "IP" },
                    { key: "state", header: "Estado", render: (row) => (row.revoked_at ? <span className="badge badge-neutral">Revocada</span> : row.current ? <span className="badge badge-ok">Actual</span> : <span className="badge badge-info">Activa</span>) },
                    {
                      key: "actions",
                      header: "",
                      render: (row) =>
                        !row.revoked_at && !row.current ? (
                          <button className="btn btn-sm btn-ghost" onClick={() => revoke(row.public_id)}>Revocar</button>
                        ) : null,
                    },
                  ]}
                />
              ) : (
                <EmptyState title="Sin sesiones registradas" />
              )}
            </CardBody>
          </Card>
        </div>

        <Card>
          <CardHeader title="Cambiar contraseña" subtitle="Política: mínimo 10 caracteres con mayúscula, minúscula, dígito y símbolo" />
          <CardBody>
            <div className="field">
              <label>Nueva contraseña</label>
              <input className="input" type="password" value={next} onChange={(event) => setNext(event.target.value)} />
            </div>
            <button className="btn btn-primary" style={{ width: "100%" }} onClick={changePassword} disabled={next.length < 10}>
              Guardar nueva contraseña
            </button>
            {user?.must_change_password ? (
              <div className="alert alert-warn" style={{ marginTop: 12 }}>
                La cuenta usa la contraseña inicial. Cámbiela para cumplir la política de seguridad institucional.
              </div>
            ) : null}
            <p className="hint" style={{ marginTop: 10 }}>
              El cambio revoca la necesidad de modificación y queda registrado en la bitácora de auditoría.
            </p>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
