## Qué cambia

<!-- Resumen del cambio y, si aplica, el requisito que cubre (M01–M16 / F1–F12). -->

## Por qué

<!-- Motivo: error corregido, requisito del protocolo, mejora de rendimiento, etc. -->

## Cómo se verificó

<!-- Comandos ejecutados y resultado. No declare nada como "funcional" sin evidencia. -->

```bash
python -m pytest backend/tests -q
cd frontend && npx tsc --noEmit && npm run build
```

- [ ] Pruebas del backend en verde
- [ ] Verificación de tipos y build del frontend
- [ ] Probado manualmente el flujo afectado (indicar cuál)
- [ ] Si tocó Docker: imagen reconstruida y `/api/v1/health` responde

## Checklist de rigor

- [ ] **Sin datos reales de pacientes**: solo agregados distrito-semana; lo de demostración se marca `is_synthetic=True`
- [ ] **Sin resultados inventados**: las métricas y capturas provienen de una ejecución reproducible
- [ ] **Sin fuga de información**: los predictores de `t` usan solo datos `≤ t`; normalización ajustada solo con entrenamiento; prueba fuera de la selección de hiperparámetros
- [ ] **Sin secretos**: no se añaden credenciales, claves ni tokens; las nuevas variables están en `.env.example`
- [ ] **Trazabilidad actualizada** en `docs/technical/matriz-trazabilidad.md` (si hay requisito nuevo)
- [ ] **Limitaciones declaradas** en el README o en el informe técnico

## Impacto

- Migraciones de base de datos: <!-- no / sí, describir -->
- Cambios incompatibles en la API:
- Rendimiento:
- Documentación actualizada:

## Notas para quien revisa

<!-- Decisiones discutibles, alternativas descartadas o partes que conviene mirar con más detalle. -->
