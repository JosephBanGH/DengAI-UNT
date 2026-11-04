# Cómo contribuir

Gracias por el interés en **SP-01 Dengue Intelligence**. Este proyecto es una plataforma de
vigilancia epidemiológica con requisitos científicos estrictos, así que las contribuciones pasan por
el mismo filtro de rigor que el código original.

## 1. Preparar el entorno

```bash
python -m pip install --target ./libs -r backend/requirements.txt
python run_api.py                      # API en http://127.0.0.1:8000

cd frontend && npm install && npm run dev
```

Pruebas antes de enviar nada:

```bash
PYTHONPATH="libs:backend:." python -m pytest backend/tests -q
cd frontend && npx tsc --noEmit && npm run build
```

## 2. Reglas que no se negocian

1. **Nunca datos reales de pacientes.** Solo agregados distrito-semana; sin identificadores
   personales. Cualquier dato de demostración debe declararse `is_synthetic=True`.
2. **Nunca resultados inventados.** No se aceptan métricas, pronósticos ni capturas que no
   provengan de una ejecución real y reproducible.
3. **Sin fuga de información.** Toda variable predictora de la semana `t` usa solo datos `≤ t`; la
   normalización se ajusta solo con entrenamiento; el conjunto de prueba no participa en la
   selección de hiperparámetros.
4. **Sin secretos en el repositorio.** Credenciales y claves solo por variables de entorno
   (`.env`, que está ignorado; documenta la nueva variable en `.env.example`).
5. **Trazabilidad.** Si agregas un requisito, actualiza
   [`docs/technical/matriz-trazabilidad.md`](docs/technical/matriz-trazabilidad.md) con su módulo,
   implementación y prueba.
6. **Honestidad sobre limitaciones.** Si algo queda parcial (una integración sin credenciales, un
   modelo no entrenado), se declara en el README y en el informe técnico.

## 3. Estilo

- **Python**: tipado en las firmas públicas, `from __future__ import annotations`, docstrings en
  español, sin lógica de negocio en los routers (va en `services/`).
- **TypeScript**: componentes funcionales, sin `any` innecesario, estilos en `styles.css` con las
  variables existentes.
- **Commits**: mensajes en imperativo y con alcance, por ejemplo
  `feat(ml): agrega validación geográfica por provincia`,
  `fix(auth): corrige expiración del refresh token`.

## 4. Flujo de trabajo

1. Abre un *issue* describiendo el problema o la mejora (hay plantillas).
2. Crea una rama: `feat/mi-mejora` o `fix/mi-correccion`.
3. Incluye pruebas que fallen sin tu cambio y pasen con él.
4. Verifica que la integración continua quede en verde.
5. Abre el *pull request* completando la plantilla: qué cambia, cómo se verificó y qué limitaciones
   permanecen.

## 5. Ámbitos donde se agradece ayuda

- Reemplazar la teselación de Voronoi por la cartografía oficial del INEI y activar PostGIS.
- Adaptadores reales para CHIRPS y WorldPop (extracción por distrito).
- Suite de pruebas de componentes del frontend (Vitest + Testing Library).
- Migrar el entrenamiento a una cola de tareas (Celery/RQ) con *workers* dedicados.
- Corrida completa de ConvLSTM y ConvLSTM-Attention con presupuesto de CPU/GPU suficiente.
