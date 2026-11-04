import { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { describeError, useToast } from "../context/ToastContext";
import { useAuth } from "../context/AuthContext";

const DEMO = [
  { email: "admin@dengai.unt.edu.pe", role: "Administrador del sistema" },
  { email: "admin.epi@dengai.unt.edu.pe", role: "Administrador epidemiológico" },
  { email: "epidemiologo@dengai.unt.edu.pe", role: "Epidemiólogo" },
  { email: "analista@dengai.unt.edu.pe", role: "Analista de datos" },
  { email: "investigador@dengai.unt.edu.pe", role: "Investigador" },
  { email: "auditor@dengai.unt.edu.pe", role: "Auditor" },
  { email: "consulta@dengai.unt.edu.pe", role: "Usuario de consulta" },
];

export function LoginPage() {
  const { user, loading, login } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();
  const [email, setEmail] = useState("admin@dengai.unt.edu.pe");
  const [password, setPassword] = useState("Admin#Dengue2025");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!loading && user) return <Navigate to="/" replace />;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const result = await login(email.trim(), password);
      toast.push("Sesión iniciada correctamente.", "success");
      if (result.mustChangePassword) {
        toast.push("Debe cambiar su contraseña inicial en Mi perfil.", "warn");
      }
      navigate("/", { replace: true });
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-hero">
        <div>
          <div className="brand-mark" style={{ marginBottom: 34 }}>
            <div className="brand-logo" style={{ width: 44, height: 44, fontSize: 17 }}>D1</div>
            <div>
              <div className="brand-name" style={{ fontSize: 15 }}>SP-01 Dengue Intelligence</div>
              <div className="brand-sub" style={{ color: "#9dc4d0" }}>Universidad Nacional de Trujillo</div>
            </div>
          </div>
          <h1>Predicción espacio-temporal y alerta temprana de brotes de dengue</h1>
          <p>
            Plataforma de apoyo a la decisión epidemiológica que integra datos climáticos, ambientales y de
            vigilancia para pronosticar casos por distrito y semana epidemiológica.
          </p>
          <ul>
            <li>Pronóstico multihorizonte con cinco algoritmos de aprendizaje profundo</li>
            <li>Clasificación de riesgo con umbrales epidemiológicos configurables</li>
            <li>Mapas territoriales, alertas tempranas y reportes PDF, DOCX y XLSX</li>
            <li>Explicabilidad SHAP y bitácora de auditoría encadenada</li>
          </ul>
        </div>
        <p style={{ fontSize: 11.5, color: "#9dc4d0" }}>
          Sistema de apoyo a la decisión. No realiza diagnóstico clínico individual ni sustituye la vigilancia
          epidemiológica oficial.
        </p>
      </div>

      <div className="login-panel">
        <div className="login-card card card-pad">
          <div className="brand-mark">
            <div className="brand-logo">D1</div>
            <div>
              <div className="brand-name" style={{ color: "var(--text)" }}>Dengue Intelligence</div>
              <div className="brand-sub" style={{ color: "var(--text-soft)" }}>Acceso institucional</div>
            </div>
          </div>
          <h2>Iniciar sesión</h2>
          <p className="muted small" style={{ marginBottom: 16 }}>
            Ingrese con las credenciales asignadas por el administrador del sistema.
          </p>
          <form onSubmit={submit}>
            <div className="field">
              <label htmlFor="email">Correo institucional</label>
              <input
                id="email"
                className="input"
                type="email"
                autoComplete="username"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>
            <div className="field">
              <label htmlFor="password">Contraseña</label>
              <input
                id="password"
                className="input"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </div>
            {error ? <div className="alert alert-error">{error}</div> : null}
            <button className="btn btn-primary" style={{ width: "100%" }} disabled={submitting}>
              {submitting ? "Verificando…" : "Ingresar"}
            </button>
          </form>

          <details className="demo-accounts">
            <summary>Credenciales de demostración</summary>
            <table className="data" style={{ fontSize: 11.5 }}>
              <thead>
                <tr>
                  <th>Rol</th>
                  <th>Correo</th>
                </tr>
              </thead>
              <tbody>
                {DEMO.map((item) => (
                  <tr key={item.email}>
                    <td>{item.role}</td>
                    <td className="mono" style={{ fontSize: 10.5 }}>{item.email}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint" style={{ marginTop: 6 }}>
              Contraseña común de demostración: <span className="mono">Demo#Dengue2025</span>. El administrador usa
              <span className="mono"> Admin#Dengue2025</span>. Cámbielas tras el primer acceso.
            </p>
          </details>
        </div>
      </div>
    </div>
  );
}
