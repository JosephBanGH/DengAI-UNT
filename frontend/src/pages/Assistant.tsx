import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { AssistantAnswer } from "../api/types";
import { PageHeader } from "../components/Layout";
import { Card, CardBody, CardHeader, EmptyState, ErrorAlert, Loading } from "../components/ui";
import { describeError } from "../context/ToastContext";

const SUGGESTIONS = [
  "¿Cómo se clasifica el riesgo epidemiológico?",
  "¿Qué métricas de validación se utilizan?",
  "¿Cuáles son los pronósticos vigentes?",
  "¿Qué fuentes de datos y licencias se usan?",
  "¿Cómo se evita la fuga de información?",
];

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: AssistantAnswer["sources"];
  provider?: string;
  latency?: number | null;
}

export function AssistantPage() {
  const [documents, setDocuments] = useState<any[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [district, setDistrict] = useState("");
  const [districts, setDistricts] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    Promise.all([api.get<any[]>("/assistant/documents"), api.get<any[]>("/geo/units")])
      .then(([docs, units]) => {
        setDocuments(docs);
        setDistricts(units);
        setReady(true);
      })
      .catch((caught) => setError(describeError(caught)));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const ask = async (text?: string) => {
    const content = (text ?? question).trim();
    if (!content || loading) return;
    setMessages((current) => [...current, { role: "user", content }]);
    setQuestion("");
    setLoading(true);
    setError(null);
    try {
      const answer = await api.post<AssistantAnswer>("/assistant/ask", {
        question: content,
        district_code: district || undefined,
        top_k: 5,
      });
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: answer.answer,
          sources: answer.sources,
          provider: answer.provider,
          latency: answer.latency_ms,
        },
      ]);
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Asistente inteligente de vigilancia"
        description="Recuperación aumentada sobre documentación interna verificada y datos de la plataforma. El asistente no genera pronósticos ni cifras: solo explica resultados existentes."
      />

      {error ? <ErrorAlert message={error} /> : null}

      <div className="grid grid-main">
        <Card style={{ display: "flex", flexDirection: "column", minHeight: 540 }}>
          <CardHeader
            title="Conversación"
            subtitle="Las respuestas citan los documentos y datos consultados"
            actions={
              <select className="select" style={{ width: 190 }} value={district} onChange={(event) => setDistrict(event.target.value)}>
                <option value="">Sin distrito específico</option>
                {districts.map((unit) => (
                  <option key={unit.code} value={unit.code}>{unit.name}</option>
                ))}
              </select>
            }
          />
          <CardBody style={{ flex: 1, overflowY: "auto", maxHeight: 460 }}>
            {!messages.length ? (
              <div>
                <EmptyState
                  title="Consultas sugeridas"
                  description="El asistente puede explicar metodología, niveles de riesgo, métricas, fuentes y pronósticos vigentes."
                />
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
                  {SUGGESTIONS.map((suggestion) => (
                    <button key={suggestion} className="chip" style={{ cursor: "pointer" }} onClick={() => ask(suggestion)}>
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} style={{ marginBottom: 14 }}>
                  <div
                    style={{
                      display: "inline-block",
                      padding: "9px 12px",
                      borderRadius: 10,
                      background: message.role === "user" ? "var(--primary-soft)" : "var(--surface-2)",
                      maxWidth: "92%",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                      {message.role === "user" ? "Consulta" : "Asistente"}
                      {message.provider ? <span className="chip" style={{ marginLeft: 8 }}>{message.provider}</span> : null}
                    </div>
                    {message.content}
                  </div>
                  {message.sources?.length ? (
                    <div className="small muted" style={{ marginTop: 6 }}>
                      Fuentes: {message.sources.map((source) => `${source.document} (${source.section})`).join(" · ")}
                    </div>
                  ) : null}
                </div>
              ))
            )}
            {loading ? <Loading label="Buscando en la documentación…" /> : null}
            <div ref={endRef} />
          </CardBody>
          <div style={{ padding: 14, borderTop: "1px solid var(--border)", display: "flex", gap: 8 }}>
            <input
              className="input"
              placeholder="Escriba su consulta sobre vigilancia, metodología o resultados…"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void ask();
              }}
            />
            <button className="btn btn-primary" onClick={() => ask()} disabled={loading || !question.trim()}>
              Enviar
            </button>
          </div>
        </Card>

        <Card>
          <CardHeader title="Documentos indexados" subtitle="Corpus verificado de la plataforma" />
          <CardBody>
            {!ready ? (
              <Loading />
            ) : (
              documents.map((document) => (
                <div key={document.doc_id} style={{ marginBottom: 12 }}>
                  <strong className="small">{document.title}</strong>
                  <div className="hint">Secciones: {document.sections.join(", ")}</div>
                </div>
              ))
            )}
            <p className="hint" style={{ marginTop: 12 }}>
              El asistente opera con un proveedor determinista y plantillado por defecto. Si se configura un
              proveedor compatible con OpenAI, sus respuestas se validan contra los hechos recuperados y se
              descartan si introducen cifras no sustentadas.
            </p>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
