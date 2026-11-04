# Política de seguridad

## Versiones soportadas

| Versión | Soporte |
|---|---|
| 1.0.x | Sí |

## Cómo reportar una vulnerabilidad

**No abra un *issue* público** con detalles de una vulnerabilidad ni con datos sensibles.

1. Escriba a la persona responsable del repositorio (ver el perfil del mantenedor en GitHub) usando
   el asunto `[SEGURIDAD] SP-01 Dengue Intelligence`.
2. Incluya: versión afectada, pasos para reproducir, impacto estimado y, si es posible, una
   propuesta de corrección.
3. Acuse de recibo en un plazo razonable; se coordinará la divulgación después de publicar el
   arreglo.

## Alcance

En alcance:

- Autenticación, sesiones y control de acceso (M01, M02).
- Fuga de información entre roles o entre usuarios.
- Inyección SQL/NoSQL, XSS, CSRF, *path traversal* en la descarga de reportes.
- Manipulación de la bitácora de auditoría (M03) sin detección.
- Exposición de secretos, credenciales o rutas internas en respuestas de error o registros.
- Ejecución de código a través de la carga de archivos (importación CSV/JSON).
- Prompt injection en el asistente (M13) que consiga operaciones no autorizadas o cifras inventadas.

Fuera de alcance:

- Ausencia de HTTPS en un despliegue local sin proxy (documentado: se espera un proxy TLS).
- Fuerza bruta masiva con credenciales de demostración sin cambiar.
- Limitaciones ya declaradas en el README (datos sintéticos, cartografía de teselación, SQLite local).

## Controles implementados

| Control | Implementación |
|---|---|
| Contraseñas | bcrypt (12 rondas), política de longitud y composición, bloqueo tras 5 intentos |
| Tokens | JWT HS256 con `iss`, `jti`, `nbf`, `exp`; refresh con hash SHA-256 y revocación por sesión |
| Autorización | RBAC con 27 permisos; el token no basta: se revalida sesión y usuario en cada petición |
| Entradas | Esquemas Pydantic; subida limitada a 20 MB; CSV/JSON validados fila a fila |
| Inyección SQL | SQLAlchemy parametrizado, sin SQL concatenado |
| Cabeceras | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, `Cache-Control: no-store` en `/api` |
| Errores | Sin trazas internas hacia el cliente; códigos estables |
| Secretos | Solo por entorno (`.env` ignorado por git; `.env.example` documenta las claves) |
| Auditoría | Bitácora encadenada por hash con verificación y exportación restringida |
| LLM | Proveedor determinista por defecto; si se usa un LLM, sus cifras se validan contra los hechos |

## Credenciales de demostración

El proyecto crea cuentas con contraseñas por defecto (`Admin#Dengue2025`, `Demo#Dengue2025`) **solo
para la demostración local**. En cualquier despliegue real:

1. Defina `BOOTSTRAP_ADMIN_PASSWORD` y `DEMO_USER_PASSWORD` con valores propios antes del primer
   arranque, o cambie las contraseñas en el primer acceso (`must_change_password = true`).
2. Defina `SECRET_KEY` con `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
3. Sirva la aplicación detrás de un proxy con TLS y no exponga PostgreSQL a Internet.
