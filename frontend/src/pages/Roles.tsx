import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { PermissionItem, RoleItem } from "../api/types";
import { PageHeader } from "../components/Layout";
import { Card, CardBody, CardHeader, EmptyState, ErrorAlert, Loading, Modal } from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function RolesPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [roles, setRoles] = useState<RoleItem[]>([]);
  const [permissions, setPermissions] = useState<PermissionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<RoleItem | null>(null);
  const [selected, setSelected] = useState<string[]>([]);

  const load = () => {
    setLoading(true);
    Promise.all([api.get<RoleItem[]>("/roles"), api.get<PermissionItem[]>("/roles/permissions/catalog")])
      .then(([roleList, permissionList]) => {
        setRoles(roleList);
        setPermissions(permissionList);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const grouped = useMemo(() => {
    const map = new Map<string, PermissionItem[]>();
    permissions.forEach((permission) => {
      const module = permission.code.split(":")[0];
      map.set(module, [...(map.get(module) || []), permission]);
    });
    return Array.from(map.entries());
  }, [permissions]);

  const save = async () => {
    if (!editing) return;
    try {
      await api.patch(`/roles/${editing.id}`, { name: editing.name, description: editing.description, permissions: selected });
      toast.push("Rol actualizado.", "success");
      setEditing(null);
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  if (loading) return <Loading label="Cargando roles y permisos…" />;

  return (
    <>
      <PageHeader
        title="Roles y permisos"
        description="Control de acceso basado en roles. Cada rol agrupa permisos por módulo funcional; los roles del sistema no pueden perder privilegios críticos."
      />

      {error ? <ErrorAlert message={error} /> : null}

      <div className="grid grid-2">
        {roles.map((role) => (
          <Card key={role.id}>
            <CardHeader
              title={role.name}
              subtitle={`${role.user_count} usuario(s) · ${role.permissions.length} permisos`}
              actions={
                can("roles:write") ? (
                  <button
                    className="btn btn-sm"
                    disabled={role.slug === "admin_sistema"}
                    onClick={() => {
                      setEditing(role);
                      setSelected(role.permissions.map((permission) => permission.code));
                    }}
                  >
                    Editar
                  </button>
                ) : null
              }
            />
            <CardBody>
              <p className="small muted" style={{ minHeight: 34 }}>{role.description}</p>
              <div className="legend">
                {role.permissions.slice(0, 10).map((permission) => (
                  <span key={permission.code} className="chip">{permission.code}</span>
                ))}
                {role.permissions.length > 10 ? <span className="chip">+{role.permissions.length - 10}</span> : null}
              </div>
            </CardBody>
          </Card>
        ))}
        {!roles.length ? <EmptyState title="Sin roles" /> : null}
      </div>

      {editing ? (
        <Modal
          title={`Editar rol · ${editing.name}`}
          onClose={() => setEditing(null)}
          footer={
            <>
              <button className="btn" onClick={() => setEditing(null)}>Cancelar</button>
              <button className="btn btn-primary" onClick={save}>Guardar permisos</button>
            </>
          }
        >
          <div className="field">
            <label>Nombre</label>
            <input className="input" value={editing.name} onChange={(event) => setEditing({ ...editing, name: event.target.value })} />
          </div>
          <div className="field">
            <label>Descripción</label>
            <textarea className="textarea" value={editing.description} onChange={(event) => setEditing({ ...editing, description: event.target.value })} />
          </div>
          <div style={{ maxHeight: 360, overflowY: "auto" }}>
            {grouped.map(([module, items]) => (
              <div key={module} style={{ marginBottom: 12 }}>
                <div className="kpi-label">{module}</div>
                {items.map((permission) => (
                  <label key={permission.code} className="small" style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "3px 0" }}>
                    <input
                      type="checkbox"
                      checked={selected.includes(permission.code)}
                      onChange={(event) => {
                        setSelected((current) =>
                          event.target.checked ? [...current, permission.code] : current.filter((code) => code !== permission.code),
                        );
                      }}
                    />
                    <span>
                      <span className="mono">{permission.code}</span>
                      <span className="hint"> — {permission.description}</span>
                    </span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        </Modal>
      ) : null}
    </>
  );
}
