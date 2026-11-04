---
name: Propuesta de mejora
about: Sugiera una funcionalidad, un modelo o una mejora metodológica
title: "[Mejora] "
labels: [enhancement, triage]
assignees: ""
---

## Necesidad

<!-- Qué problema resuelve. Si viene del protocolo SP-01, indique el requisito o la fase (F1–F12). -->

## Propuesta

<!-- Descripción de la solución esperada, sin entrar todavía en la implementación. -->

## Rigor científico

- ¿Afecta la unidad de análisis (distrito-semana) o las particiones de validación?
- ¿Introduce riesgo de fuga de información?
- ¿Requiere datos que no están disponibles o autorizados?
- ¿Cómo se mediría que aporta valor (métrica y prueba estadística)?

## Alternativas consideradas

<!-- Otras opciones y por qué se descartan. -->

## Alcance

- Módulos afectados:
- ¿Rompe compatibilidad con la API o con la base de datos?
- ¿Requiere migración (Alembic)?

## Comprobaciones

- [ ] No incluye datos reales de pacientes ni secretos.
- [ ] Declaro las limitaciones si la propuesta queda parcial.
