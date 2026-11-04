import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Page, RoleItem, UserItem } from "../api/types";
import { PageHeader } from "../components/Layout";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorAlert,
  Loading,
  Modal,
  StatusBadge,
  formatDate,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

const EMPTY_FORM = {
  email: "",
  full_name: "",
  password: "",
  position: "",
  organization: "",
  roles: ["consulta"],
};

export function UsersPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [users, setUsers] = useState<UserItem[]>([]);
  const [roles, setRoles] = useState<RoleItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [editing, setEditing] = useState<UserItem | null>(null);

  const load = () => {
    setLoading(true);
    Promise.all([
      api.get<Page<UserItem>>("/users", { size: 100, search: search || undefined }),
      api.get<RoleItem[]>("/roles"),
    ])
      .then(([userPage, roleList]) => {
        setUsers(userPage.items);
        setRoles(roleList);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const create = async () => {
    try {
      await api.post("/users", form);
      toast.push("Usuario creado. Debe cambiar la contraseña en su primer acceso.", "success");
      setCreating(false);
      setForm({ ...EMPTY_FORM });
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const toggleStatus = async (user: UserItem) => {
    try {
      await api.post(`/users/${user.id}/status`, { is_active: !user.is_active, reason: "cambio administrativo" });
      toast.push(`Usuario ${user.is_active ? "desactivado" : "activado"}.`, "success");
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const saveRoles = async () => {
    if (!editing) return;
    try {
      await api.patch(`/users/${editing.id}`, {
        full_name: editing.full_name,
        position: editing.position,
        organization: editing.organization,
        roles: editing.roles.map((role) => role.slug),
      });
      toast.push("Usuario actualizado.", "success");
      setEditing(null);
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  if (loading) return <Loading label="Cargando usuarios…" />;

  return (
    <>
      <PageHeader
        title="Administración de usuarios"
        description="Alta, modificación, activación y asignación de roles. El control de acceso se basa en permisos (RBAC) con principio de mínimo privilegio."
        actions={
          can("users:write") ? (
            <button className="btn btn-primary" onClick={() => setCreating(true)}>Nuevo usuario</button>
          ) : undefined
        }
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Card>
        <CardHeader
          title="Usuarios registrados"
          subtitle={`${users.length} cuentas`}
          actions={
            <div style={{ display: "flex", gap: 8 }}>
              <input className="input" placeholder="Buscar por nombre o correo" value={search} onChange={(event) => setSearch(event.target.value)} />
              <button className="btn" onClick={load}>Buscar</button>
            </div>
          }
        />
        <CardBody>
          <DataTable
            rows={users}
            rowKey={(row) => row.id}
            columns={[
              { key: "full_name", header: "Usuario", render: (row) => (
                <div>
                  <strong>{row.full_name}</strong>
                  <div className="hint">{row.email}</div>
                </div>
              ) },
              { key: "roles", header: "Roles", render: (row) => (
                <div className="legend">
                  {row.roles.map((role) => <span key={role.slug} className="chip">{role.name}</span>)}
                </div>
              ) },
              { key: "position", header: "Cargo", render: (row) => <span className="small">{row.position || "—"}</span> },
              { key: "is_active", header: "Estado", render: (row) => (
                <span className={`badge ${row.is_active ? "badge-ok" : "badge-danger"}`}>
                  {row.is_active ? "Activo" : "Inactivo"}
                </span>
              ) },
              { key: "must_change_password", header: "Clave inicial", render: (row) => (row.must_change_password ? <span className="badge badge-warn">Pendiente</span> : <span className="muted small">Actualizada</span>) },
              { key: "last_login_at", header: "Último acceso", render: (row) => formatDate(row.last_login_at, true) },
              {
                key: "actions",
                header: "",
                render: (row) =>
                  can("users:write") ? (
                    <div style={{ display: "flex", gap: 6 }}>
                      <button className="btn btn-sm btn-ghost" onClick={() => setEditing(row)}>Editar</button>
                      <button className="btn btn-sm btn-ghost" onClick={() => toggleStatus(row)}>
                        {row.is_active ? "Desactivar" : "Activar"}
                      </button>
                    </div>
                  ) : null,
              },
            ]}
          />
        </CardBody>
      </Card>

      {creating ? (
        <Modal
          title="Nuevo usuario"
          onClose={() => setCreating(false)}
          footer={
            <>
              <button className="btn" onClick={() => setCreating(false)}>Cancelar</button>
              <button className="btn btn-primary" onClick={create}>Crear usuario</button>
            </>
          }
        >
          <div className="field">
            <label>Correo</label>
            <input className="input" type="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
          </div>
          <div className="field">
            <label>Nombre completo</label>
            <input className="input" value={form.full_name} onChange={(event) => setForm({ ...form, full_name: event.target.value })} />
          </div>
          <div className="form-row">
            <div className="field">
              <label>Cargo</label>
              <input className="input" value={form.position} onChange={(event) => setForm({ ...form, position: event.target.value })} />
            </div>
            <div className="field">
              <label>Organización</label>
              <input className="input" value={form.organization} onChange={(event) => setForm({ ...form, organization: event.target.value })} />
            </div>
          </div>
          <div className="field">
            <label>Contraseña inicial</label>
            <input className="input" type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />
            <span className="hint">Mínimo 10 caracteres con mayúscula, minúscula, dígito y carácter especial.</span>
          </div>
          <div className="field">
            <label>Rol</label>
            <select className="select" value={form.roles[0]} onChange={(event) => setForm({ ...form, roles: [event.target.value] })}>
              {roles.map((role) => <option key={role.slug} value={role.slug}>{role.name}</option>)}
            </select>
          </div>
        </Modal>
      ) : null}

      {editing ? (
        <Modal
          title={`Editar ${editing.full_name}`}
          onClose={() => setEditing(null)}
          footer={
            <>
              <button className="btn" onClick={() => setEditing(null)}>Cancelar</button>
              <button className="btn btn-primary" onClick={saveRoles}>Guardar</button>
            </>
          }
        >
          <div className="field">
            <label>Nombre completo</label>
            <input className="input" value={editing.full_name} onChange={(event) => setEditing({ ...editing, full_name: event.target.value })} />
          </div>
          <div className="form-row">
            <div className="field">
              <label>Cargo</label>
              <input className="input" value={editing.position} onChange={(event) => setEditing({ ...editing, position: event.target.value })} />
            </div>
            <div className="field">
              <label>Organización</label>
              <input className="input" value={editing.organization} onChange={(event) => setEditing({ ...editing, organization: event.target.value })} />
            </div>
          </div>
          <div className="field">
            <label>Roles</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {roles.map((role) => {
                const assigned = editing.roles.some((item) => item.slug === role.slug);
                return (
                  <label key={role.slug} className="small" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={assigned}
                      onChange={(event) => {
                        const next = event.target.checked
                          ? [...editing.roles, { id: role.id, slug: role.slug, name: role.name }]
                          : editing.roles.filter((item) => item.slug !== role.slug);
                        setEditing({ ...editing, roles: next });
                      }}
                    />
                    {role.name} <span className="hint">({role.permissions.length} permisos)</span>
                  </label>
                );
              })}
            </div>
          </div>
        </Modal>
      ) : null}
    </>
  );
}
