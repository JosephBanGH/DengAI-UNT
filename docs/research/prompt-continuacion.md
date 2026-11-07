# Prompt de continuación — SP-01 Dengue Intelligence

> **Cuándo usarlo:** solo cuando abras una **sesión nueva** (o el agente perdió el contexto).
> No lo uses dentro de la misma sesión: ahí basta con responder «continúa».
>
> **No vuelvas a pegar el prompt maestro completo** en una sesión de continuación: le haría
> repetir la Fase 0 y podrías duplicar trabajo.

---

## Plantilla (copiar y pegar)

```text
Continúo el proyecto DengAI-UNT. Antes de actuar, lee en este orden:

1. docs/research/prompt-maestro-v2.md
2. docs/CasoSP01.docx (incluye el ANEXO del asesor al final)
3. docs/research/protocolo-experimental.md
4. docs/technical/matriz-trazabilidad.md
5. README.md
6. Ejecuta: git log --oneline -10 y git status

Luego:

- Reporta qué fases (F0–F7) están realmente completadas, con evidencia verificable:
  archivos, pruebas ejecutadas, endpoints, checksums, contenedores levantados.
- Distingue explícitamente lo verificado de lo declarado pero no comprobado.
- Identifica la primera fase incompleta o con evidencia insuficiente.
- Continúa desde esa fase, sin repetir trabajo ya verificado.
- No inventes datos, métricas ni integraciones. Lo no verificable se declara como limitación.
- Al final de cada fase, entrega el reporte: qué se hizo, archivos, comandos, evidencia,
  limitaciones y siguiente paso.
```

---

## Variante: auditoría antes de continuar

Úsala si sospechas que la sesión anterior declaró cosas sin verificar.

```text
Antes de continuar con el proyecto DengAI-UNT, audita el estado real.

Lee docs/research/prompt-maestro-v2.md, docs/CasoSP01.docx (anexo),
docs/research/protocolo-experimental.md y docs/technical/matriz-trazabilidad.md.
Revisa el código y ejecuta las pruebas existentes.

Para cada afirmación de "implementado" en la documentación, comprueba:
- ¿Existe el archivo?
- ¿Se ejecuta sin error?
- ¿Hay una prueba que lo respalde?
- ¿El dato es real, sintético o una plantilla de contingencia?

Entrega una tabla: requisito | estado real | evidencia | brecha.
No propongas cambios cosméticos. Si no hay defectos, dilo explícitamente.
Después de la auditoría, continúa desde la primera fase incompleta.
```

---

## Regla de oro

| Situación | Acción |
|---|---|
| La sesión sigue viva y recuerda el trabajo | Responde «continúa» |
| Sesión cerrada o contexto perdido | Usa la plantilla de este archivo |
| Dudas de que lo declarado sea real | Usa la variante de auditoría |
| Cambias de fase manualmente | Solo indica «continúa desde la Fase N» |
