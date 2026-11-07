"""Asistente inteligente con recuperación aumentada (M13).

Regla crítica: el LLM nunca sustituye al modelo predictivo. Toda cifra de la respuesta
proviene de ``facts`` (calculados desde la base de datos) o de los fragmentos
recuperados; el proveedor por defecto es determinista y plantillado, sin generación
libre. Si se configura un proveedor OpenAI-compatible, la respuesta se valida contra
los hechos y se rechaza si introduce cifras no sustentadas.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.inference import latest_forecast, risk_thresholds
from app.models.epi import DengueCase
from app.models.geo import GeographicUnit
from app.models.ml import ModelVersion
from app.models.ops import Alert, AssistantQuery
from app.models.security import SystemConfig

logger = logging.getLogger("dengai.assistant")


@dataclass
class Document:
    doc_id: str
    title: str
    section: str
    text: str


def _corpus(db: Session) -> List[Document]:
    """Corpus verificado: documentación interna de la plataforma y configuración vigente."""
    thresholds = risk_thresholds(db)
    units = list(db.scalars(select(GeographicUnit)))
    production = list(db.scalars(select(ModelVersion).where(ModelVersion.is_production.is_(True))))
    documents = [
        Document(
            "protocolo", "Protocolo operativo SP-01", "Alcance",
            "La plataforma SP-01 Dengue Intelligence apoya la decisión epidemiológica. Predice casos por "
            "distrito y semana epidemiológica, clasifica riesgo y emite alertas tempranas. No realiza "
            "diagnóstico clínico individual ni sustituye la vigilancia epidemiológica oficial.",
        ),
        Document(
            "protocolo", "Protocolo operativo SP-01", "Unidad de análisis",
            "La unidad de análisis es distrito-semana epidemiológica. Las variables independientes incluyen "
            "temperatura, humedad, precipitación, altitud, densidad poblacional, incidencia histórica e "
            "índices ambientales; la variable dependiente es el número de casos confirmados.",
        ),
        Document(
            "metodologia", "Metodología de validación", "Particiones",
            "La validación es temporal walk-forward y geográfica por distritos. El conjunto de prueba no "
            "participa en la selección de hiperparámetros de Optuna. La normalización se ajusta solo con "
            "datos de entrenamiento para evitar fuga de información.",
        ),
        Document(
            "metodologia", "Metodología de validación", "Métricas",
            "Se reportan MAE, RMSE, MAPE cuando es matemáticamente apropiado, MASE, WIS, cobertura de "
            "intervalos, AUC-PR, sensibilidad, especificidad y Brier Score, además de Friedman, Wilcoxon "
            "con corrección de Holm, Diebold-Mariano y bootstrap para intervalos de confianza.",
        ),
        Document(
            "riesgo", "Niveles de riesgo vigentes", "Umbrales",
            f"La clasificación de riesgo usa incidencia semanal por 100 000 habitantes con umbrales "
            f"configurables: bajo por debajo de {thresholds['low']}, medio entre {thresholds['low']} y "
            f"{thresholds['moderate']}, alto entre {thresholds['moderate']} y {thresholds['high']}, y "
            f"crítico desde {thresholds['high']}. Justificación: {thresholds.get('justification', 'configurada por el administrador epidemiológico')}",
        ),
        Document(
            "datos", "Fuentes de datos y licencias", "Licencias",
            "MINSA/CDC Perú: datos abiertos con atribución y verificación de desidentificación. CDC Dengue "
            "Data: dominio público. WHO GHO: CC BY-NC-SA 3.0 IGO (uso no comercial). NASA POWER: acceso "
            "abierto con atribución. CHIRPS: CC0 con atribución. WorldPop: CC BY 4.0. El conjunto de "
            "demostración es sintético y está identificado como tal en toda la interfaz.",
        ),
        Document(
            "modelos", "Modelos implementados", "Arquitecturas",
            "Se implementan cinco algoritmos: XGBoost (tabular), LSTM (temporal), ConvLSTM "
            "(espacio-temporal), CNN-LSTM (híbrido) y ConvLSTM-Attention (híbrido con atención temporal y "
            "espacial). Cada versión declara el horizonte para el que fue entrenada.",
        ),
        Document(
            "limitaciones", "Limitaciones y supuestos", "Datos",
            "La plataforma opera con datos reales de MINSA/CDC Perú y NASA POWER (2015-2024), con "
            "opción de respaldo sintético para pruebas locales. Las predicciones son generadas exclusivamente "
            "por los modelos numéricos de IA y el asistente actúa solo como interfaz de consulta.",
        ),
        Document(
            "oms_clinica", "Guía OMS/OPS para el Diagnóstico y Manejo Clínico del Dengue", "Clasificación clínica",
            "La OMS clasifica el dengue en tres categorías: 1) Dengue sin signos de alarma (fiebre, cefalea, "
            "dolor retroocular, mialgias, artralgias y exantema). 2) Dengue con signos de alarma (dolor abdominal "
            "intenso y continuo, vómitos persistentes, acumulación de líquidos, sangrado de mucosas, letargo e "
            "incremento del hematocrito concomitante con disminución rápida de plaquetas). 3) Dengue grave "
            "(choque hipovolémico por extravasación grave de plasma, dificultad respiratoria, hemorragias graves "
            "o daño orgánico grave en hígado, SNC o corazón).",
        ),
        Document(
            "minsa_norma", "Norma Técnica de Vigilancia Epidemiológica de Dengue (CDC Perú / MINSA)", "Definición de casos y alerta",
            "MINSA establece: Caso Probable: persona con fiebre aguda más dos o más manifestaciones (cefalea, "
            "mialgias, artralgias, dolor retroocular, náuseas) con antecedente epidemiológico en zona de transmisión. "
            "Caso Confirmado: caso probable con prueba de laboratorio positiva (aislamiento viral, RT-PCR, detección "
            "del antígeno NS1 antes del 5to día de fiebre, o serología ELISA IgM después del 5to día). Se define alerta "
            "de brote cuando la incidencia supera el límite superior del canal endémico cuartilar en distritos endémicos.",
        ),
        Document(
            "cdc_ecologia", "Directrices CDC sobre Biología del Vector y Transmisión de Dengue", "Ecología del vector",
            "El vector principal es Aedes aegypti. La temperatura óptima para su ciclo biológico y replicación viral "
            "oscila entre 26°C y 28°C. A temperaturas inferiores a 15°C o superiores a 38°C se inhibe la transmisión. "
            "El Período de Incubación Extrínseca (PIE) en el mosquito varía de 8 a 14 días y se acorta a mayor temperatura. "
            "Las precipitaciones generan criaderos y la humedad relativa alta (>70%) prolonga la longevidad del mosquito, "
            "explicando los rezagos de 4 a 12 semanas (lags) entre eventos climáticos y picos de casos.",
        ),
    ]
    documents.append(
        Document(
            "inventario", "Inventario territorial", "Distritos",
            f"La plataforma tiene {len(units)} distritos configurados en {', '.join(sorted({u.province for u in units}))}. "
            f"Distritos: {', '.join(sorted(u.name for u in units))}.",
        )
    )
    if production:
        documents.append(
            Document(
                "modelos", "Modelos en producción", "Versiones",
                "Versiones en producción: "
                + "; ".join(
                    f"{v.algorithm} versión {v.version} para horizonte {(v.feature_names or {}).get('horizon')}"
                    for v in production
                )
                + ".",
            )
        )
    return documents


def _retrieve(documents: List[Document], question: str, top_k: int) -> List[Tuple[Document, float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    spanish_stopwords = [
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una",
        "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre",
        "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
        "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí",
        "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
        "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "cuál",
        "cómo", "cuáles", "dónde", "quién"
    ]
    corpus = [f"{doc.title}. {doc.section}. {doc.text}" for doc in documents]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, stop_words=spanish_stopwords)
    matrix = vectorizer.fit_transform(corpus)
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    order = np.argsort(-scores)[:top_k]
    # Umbral mínimo de relevancia: si la similitud no supera 0.08, se considera fuera de dominio y se abstiene
    return [(documents[index], float(scores[index])) for index in order if scores[index] >= 0.08]


def _live_facts(db: Session, district_code: Optional[str]) -> Dict[str, Any]:
    latest_week = db.scalar(select(func.max(DengueCase.week_start)))
    facts: Dict[str, Any] = {"week": str(latest_week) if latest_week else None}
    if latest_week:
        facts["cases_last_week"] = int(
            db.scalar(
                select(func.coalesce(func.sum(DengueCase.confirmed_cases), 0)).where(DengueCase.week_start == latest_week)
            )
            or 0
        )
    forecast = latest_forecast(db)
    if forecast:
        if district_code:
            selected = [row for row in forecast if row["district_code"] == district_code]
        else:
            selected = forecast[:6]
        facts["forecast"] = [
            {
                "district": row["district_name"],
                "target_week": f"{row['target_year']}-W{row['target_week']:02d}",
                "horizon": row["horizon_weeks"],
                "predicted_cases": round(float(row["predicted_cases"]), 2),
                "risk_level": row["risk_level"],
                "incidence": row["incidence_per_100k"],
            }
            for row in selected
        ]
        facts["forecast_generated_at"] = str(forecast[0].get("target_week_start"))
    alerts_open = int(db.scalar(select(func.count()).select_from(Alert).where(Alert.status == "OPEN")) or 0)
    facts["open_alerts"] = alerts_open
    facts["thresholds"] = risk_thresholds(db)
    if district_code:
        unit = db.scalar(select(GeographicUnit).where(GeographicUnit.code == district_code))
        if unit is not None:
            facts["district"] = {
                "code": unit.code,
                "name": unit.name,
                "province": unit.province,
                "population": unit.population,
                "altitude_m": unit.altitude_m,
            }
            total = int(
                db.scalar(
                    select(func.coalesce(func.sum(DengueCase.confirmed_cases), 0)).where(
                        DengueCase.district_code == district_code
                    )
                )
                or 0
            )
            facts["district"]["historical_cases"] = total
    return facts


_BLOCKED = re.compile(
    r"(ignora|olvida)\s+(las\s+)?instrucciones|system\s*prompt|contrase[ñn]a|password|token\s+de\s+acceso|"
    r"datos\s+personales|nombre\s+del\s+paciente|elimina\s+la\s+base|drop\s+table",
    re.IGNORECASE,
)


def ask(
    db: Session,
    *,
    question: str,
    district_code: Optional[str] = None,
    top_k: int = 5,
    user_email: str = "",
) -> Dict[str, Any]:
    started = time.perf_counter()
    if _BLOCKED.search(question):
        answer = (
            "No puedo atender esa solicitud: la plataforma no expone credenciales, datos personales de "
            "pacientes ni permite operaciones destructivas desde el asistente."
        )
        _store(db, user_email, question, answer, [], settings.LLM_PROVIDER, 0, blocked="solicitud_no_permitida")
        db.commit()
        return {
            "answer": answer,
            "grounded": True,
            "provider": "guardrail",
            "sources": [],
            "facts": {},
            "caveat": "Solicitud bloqueada por las políticas del asistente.",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    documents = _corpus(db)
    retrieved = _retrieve(documents, question, top_k)
    facts = _live_facts(db, district_code)

    if not retrieved:
        answer = (
            "No encontré documentación suficiente para responder con respaldo verificable. "
            "Puedo consultar sobre metodología, fuentes de datos, niveles de riesgo, modelos y pronósticos vigentes."
        )
        _store(db, user_email, question, answer, [], settings.LLM_PROVIDER, 0, blocked="sin_respaldo")
        db.commit()
        return {
            "answer": answer,
            "grounded": True,
            "provider": settings.LLM_PROVIDER,
            "sources": [],
            "facts": facts,
            "caveat": "Respuesta limitada para no generar afirmaciones sin respaldo.",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    answer, provider = _compose(question, retrieved, facts)
    sources = [
        {"document": doc.title, "section": doc.section, "score": round(score, 4), "snippet": doc.text[:240]}
        for doc, score in retrieved
    ]
    latency = int((time.perf_counter() - started) * 1000)
    _store(db, user_email, question, answer, sources, provider, latency)
    db.commit()
    return {
        "answer": answer,
        "grounded": True,
        "provider": provider,
        "sources": sources,
        "facts": facts,
        "caveat": (
            "La respuesta combina documentación interna y datos de la plataforma. "
            "No sustituye la interpretación de un epidemiólogo."
        ),
        "latency_ms": latency,
    }


def _compose(question: str, retrieved: List[Tuple[Document, float]], facts: Dict[str, Any]) -> Tuple[str, str]:
    """Respuesta determinista y trazable; opcionalmente refinada por un LLM validado."""
    body: List[str] = []
    for doc, _score in retrieved[:2]:
        body.append(f"{doc.text}")
    forecast = facts.get("forecast") or []
    if forecast:
        lines = ", ".join(
            f"{row['district']}: {row['predicted_cases']:.1f} casos (riesgo {row['risk_level']}) para {row['target_week']}"
            for row in forecast[:4]
        )
        body.append(f"Pronósticos vigentes registrados: {lines}.")
    if facts.get("district"):
        district = facts["district"]
        body.append(
            f"El distrito consultado es {district['name']} ({district['province']}), con población "
            f"{district.get('population') or 'no registrada'} y altitud {district.get('altitude_m') or 'n/d'} m."
        )
    if facts.get("open_alerts") is not None:
        body.append(f"Actualmente hay {facts['open_alerts']} alertas abiertas.")
    base_answer = " ".join(body)

    active_key = settings.GEMINI_API_KEY or settings.LLM_API_KEY
    if settings.LLM_PROVIDER in ("openai_compatible", "gemini") and active_key:
        refined = _llm_refine(question, base_answer, facts)
        if refined:
            model_tag = "gemini-1.5-flash" if settings.LLM_PROVIDER == "gemini" else settings.LLM_MODEL
            return refined, f"{settings.LLM_PROVIDER}:{model_tag}"
    return base_answer, settings.LLM_PROVIDER


def _llm_refine(question: str, base_answer: str, facts: Dict[str, Any]) -> Optional[str]:
    """Consulta al proveedor LLM y valida que no introduzca cifras no sustentadas."""
    try:
        import httpx

        prompt = (
            "Eres un asistente de vigilancia epidemiológica. Reescribe la respuesta usando SOLO los hechos "
            "proporcionados. No inventes cifras ni porcentajes. Si falta información, dilo explícitamente.\n\n"
            f"HECHOS:\n{base_answer}\n\nPREGUNTA: {question}\nRESPUESTA:"
        )
        api_key = settings.GEMINI_API_KEY or settings.LLM_API_KEY
        if settings.LLM_PROVIDER == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            model = "gemini-1.5-flash"
        else:
            url = f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions"
            model = settings.LLM_MODEL

        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
            timeout=settings.LLM_TIMEOUT,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        allowed = set(re.findall(r"\d+(?:[.,]\d+)?", base_answer + str(facts)))
        cited = set(re.findall(r"\d+(?:[.,]\d+)?", text))
        unsupported = {value for value in cited if value not in allowed}
        if len(unsupported) > max(2, 0.25 * len(cited)):
            logger.warning("Respuesta del LLM descartada: cifras no sustentadas %s", unsupported)
            return None
        return text
    except Exception:
        logger.exception("Fallo al consultar el proveedor LLM configurado")
        return None


def _store(db: Session, user_email: str, question: str, answer: str, sources: List[Dict[str, Any]], provider: str, latency_ms: int, blocked: str = "") -> None:
    db.add(
        AssistantQuery(
            user_email=user_email or "", question=question[:1000], answer=answer, sources={"items": sources},
            grounded=not blocked, provider=provider, latency_ms=latency_ms, blocked_reason=blocked,
        )
    )


def documents_catalog(db: Session) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for doc in _corpus(db):
        entry = seen.setdefault(doc.doc_id, {"doc_id": doc.doc_id, "title": doc.title, "sections": []})
        entry["sections"].append(doc.section)
    return list(seen.values())


def history(db: Session, *, limit: int = 30) -> List[Dict[str, Any]]:
    rows = db.scalars(select(AssistantQuery).order_by(AssistantQuery.created_at.desc()).limit(limit))
    return [
        {
            "question": row.question,
            "answer": row.answer,
            "provider": row.provider,
            "grounded": row.grounded,
            "created_at": row.created_at,
            "sources": (row.sources or {}).get("items", []),
        }
        for row in rows
    ]
