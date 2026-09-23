import {
  type CSSProperties,
  type FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import ApiInspector, { type ApiLogEntry } from "./ApiInspector";
import { DIMENSIONS, DIMENSION_KEYS, type DimensionKey } from "./dimensions";
import type { ToneAttempt, ToneResponse } from "./types";

const MAX_SOURCE_LENGTH = 240;
const SESSION_STORAGE_KEY = "tone-jev-session";
const SESSION_PATTERN = /^[A-Za-z0-9_-]{16,64}$/;
const MAX_API_LOG_ENTRIES = 20;
const DEFAULT_SOURCE = "Please read the manual.";
const STARTING_TEXTS = [
  DEFAULT_SOURCE,
  "The meeting starts at nine.",
  "Could you put the bins out?",
  "We need to talk about the spreadsheet.",
  "Your parcel is behind the shed.",
  "I have updated the shared document.",
] as const;

const sessionId = (): string => {
  const stored = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (stored && SESSION_PATTERN.test(stored)) return stored;

  const generated =
    typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `tone-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  window.localStorage.setItem(SESSION_STORAGE_KEY, generated);
  return generated;
};

const scoreText = (value: number): string =>
  Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);

const verdictText = (result: ToneResponse): string => {
  if (result.hit) return "hit";
  if (result.distance <= 15) return "near miss";
  if (result.distance <= 40) return "not quite";
  return "way off";
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isPercentage = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 100;

const isToneAttempt = (value: unknown): value is ToneAttempt =>
  isRecord(value) &&
  typeof value.phrase === "string" &&
  Boolean(value.phrase.trim()) &&
  value.phrase.length <= MAX_SOURCE_LENGTH &&
  isPercentage(value.score) &&
  (value.confidence === null || typeof value.confidence === "number");

const readNdjson = async (
  response: Response,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<Record<string, unknown>[]> => {
  if (!response.body) throw new Error("The tone stream was empty");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const events: Record<string, unknown>[] = [];
  let buffer = "";

  const consume = (line: string) => {
    if (!line.trim()) return;
    const event: unknown = JSON.parse(line);
    if (!isRecord(event)) throw new Error("The tone stream was invalid");
    events.push(event);
    onEvent(event);
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    lines.forEach(consume);
    if (done) break;
  }
  consume(buffer);
  return events;
};

const isToneResponse = (value: unknown): value is ToneResponse => {
  if (!isRecord(value) || !Array.isArray(value.attempts)) return false;
  if (
    typeof value.phrase !== "string" ||
    !value.phrase.trim() ||
    value.phrase.length > MAX_SOURCE_LENGTH ||
    typeof value.dimension !== "string" ||
    !(value.dimension in DIMENSIONS) ||
    !isPercentage(value.target) ||
    !isPercentage(value.score) ||
    !isPercentage(value.distance) ||
    typeof value.hit !== "boolean" ||
    value.attempts.length < 1 ||
    value.attempts.length > 4
  ) {
    return false;
  }

  return value.attempts.every(
    (attempt) =>
      isRecord(attempt) &&
      typeof attempt.phrase === "string" &&
      Boolean(attempt.phrase.trim()) &&
      attempt.phrase.length <= MAX_SOURCE_LENGTH &&
      isPercentage(attempt.score) &&
      (attempt.confidence === null ||
        (typeof attempt.confidence === "number" &&
          Number.isFinite(attempt.confidence))),
  );
};

export default function App() {
  const [source, setSource] = useState(DEFAULT_SOURCE);
  const [dimensionKey, setDimensionKey] = useState<DimensionKey>("panic");
  const [target, setTarget] = useState(60);
  const [result, setResult] = useState<ToneResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(
    "Set the dial. Granite writes; Jev judges.",
  );
  const [error, setError] = useState<string | null>(null);
  const [apiInspectorOpen, setApiInspectorOpen] = useState(
    () => window.location.hash === "#inspect-api",
  );
  const [apiLog, setApiLog] = useState<ApiLogEntry[]>([]);
  const [streamAttempts, setStreamAttempts] = useState<ToneAttempt[]>([]);
  const sourceRef = useRef<HTMLTextAreaElement>(null);
  const inspectApiLinkRef = useRef<HTMLAnchorElement>(null);
  const apiRequestIdRef = useRef(0);
  const dimension = DIMENSIONS[dimensionKey];
  const sortedDimensions = useMemo(
    () =>
      [...DIMENSION_KEYS].sort((left, right) =>
        DIMENSIONS[left].name.localeCompare(DIMENSIONS[right].name),
      ),
    [],
  );

  useEffect(() => {
    const syncInspectorToHash = () => {
      setApiInspectorOpen(window.location.hash === "#inspect-api");
    };
    window.addEventListener("hashchange", syncInspectorToHash);
    return () => window.removeEventListener("hashchange", syncInspectorToHash);
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (loading) return;

    setLoading(true);
    setError(null);
    setStreamAttempts([]);
    setMessage("Granite writes. Jev judges. The loop tightens.");
    const requestBody = { source, dimension: dimensionKey, target };
    const requestId = ++apiRequestIdRef.current;
    setApiLog((current) => [
      ...current.slice(-(MAX_API_LOG_ENTRIES - 1)),
      { id: requestId, request: requestBody, status: "pending" },
    ]);
    let responseStatus: number | undefined;

    try {
      const response = await fetch("/api/tone", {
        method: "POST",
        headers: {
          accept: "application/x-ndjson",
          "content-type": "application/json",
          "x-tone-session": sessionId(),
        },
        body: JSON.stringify(requestBody),
      });
      responseStatus = response.status;
      let body: unknown;
      let inspectedResponse: unknown;
      if (
        response.headers.get("content-type")?.includes("application/x-ndjson")
      ) {
        let finalEvent: Record<string, unknown> | undefined;
        const events = await readNdjson(response, (streamEvent) => {
          if (
            streamEvent.type === "attempt" &&
            isToneAttempt(streamEvent.attempt)
          ) {
            setStreamAttempts((current) => [
              ...current,
              streamEvent.attempt as ToneAttempt,
            ]);
          }
          if (streamEvent.type === "complete" || streamEvent.type === "error") {
            finalEvent = streamEvent;
          }
        });
        inspectedResponse = events;
        if (!finalEvent) throw new Error("The tone stream ended early");
        responseStatus =
          typeof finalEvent.status === "number" ? finalEvent.status : 502;
        body =
          finalEvent.type === "complete"
            ? finalEvent.result
            : {
                error:
                  typeof finalEvent.error === "string"
                    ? finalEvent.error
                    : "The tone loop lost the plot",
              };
      } else {
        try {
          body = await response.json();
          inspectedResponse = body;
        } catch {
          setApiLog((current) =>
            current.map((entry) =>
              entry.id === requestId
                ? {
                    ...entry,
                    httpStatus: response.status,
                    response: { error: "Response was not valid JSON." },
                    status: "error",
                  }
                : entry,
            ),
          );
          throw new Error("The tone loop returned invalid JSON");
        }
      }
      const finalStatus = responseStatus ?? response.status;
      const inspection = isRecord(body) ? body.inspection : undefined;
      const modelCalls = isRecord(inspection)
        ? inspection.model_calls
        : undefined;
      setApiLog((current) =>
        current.map((entry) =>
          entry.id === requestId
            ? {
                ...entry,
                httpStatus: finalStatus,
                response: inspectedResponse,
                modelCalls: Array.isArray(modelCalls) ? modelCalls : undefined,
                status:
                  finalStatus >= 200 &&
                  finalStatus < 300 &&
                  isToneResponse(body)
                    ? "complete"
                    : "error",
              }
            : entry,
        ),
      );
      if (finalStatus < 200 || finalStatus >= 300) {
        const responseError =
          isRecord(body) && typeof body.error === "string" ? body.error : null;
        throw new Error(responseError || "The tone loop lost the plot");
      }
      if (!isToneResponse(body)) {
        throw new Error("The tone loop returned an invalid result");
      }

      setResult(body);
      setMessage(
        body.hit
          ? "Close enough for Jev. Human judgement still applies."
          : "Closest attempt kept. The dial and the judge disagreed.",
      );
    } catch (caught) {
      setApiLog((current) =>
        current.map((entry) =>
          entry.id === requestId && entry.status === "pending"
            ? {
                ...entry,
                httpStatus: responseStatus,
                response: { error: "Network request failed." },
                status: "error",
              }
            : entry,
        ),
      );
      const nextError =
        caught instanceof Error
          ? caught.message
          : "The tone loop lost the plot";
      setError(nextError);
      setMessage("Nothing was replaced. Adjust the dial or try again.");
    } finally {
      setLoading(false);
    }
  };

  const copyResult = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.phrase);
      setMessage("Copied.");
    } catch {
      setMessage("Copy failed. Select the result instead.");
    }
  };

  const reuseResult = () => {
    if (!result) return;
    setSource(result.phrase);
    setMessage("Result loaded as the next starting text.");
    sourceRef.current?.focus();
  };

  const pickRandomSource = () => {
    const choices = STARTING_TEXTS.filter((value) => value !== source);
    setSource(
      choices[Math.floor(Math.random() * choices.length)] ?? DEFAULT_SOURCE,
    );
    setMessage("Fresh starting text.");
    sourceRef.current?.focus();
  };

  const closeApiInspector = () => {
    setApiInspectorOpen(false);
    if (window.location.hash === "#inspect-api") {
      window.history.replaceState(
        null,
        "",
        `${window.location.pathname}${window.location.search}`,
      );
    }
    inspectApiLinkRef.current?.focus();
  };

  return (
    <main className="tool-shell">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="tone your mind home">
          tone your mind<span>.</span>
        </a>
        <a className="sibling-link" href="https://jev-tone.tomv.uk">
          play the original <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="kicker">choose → written live → adjust</p>
        <h1 id="page-title">
          Pick a tone.
          <span>Let the machine write.</span>
        </h1>
        <p className="lede">
          Need to panic someone, but only 60%? Pick a feeling, turn the dial,
          and let one AI rewrite while another marks its homework.
        </p>
      </section>

      <form className="tone-card" onSubmit={submit} aria-busy={loading}>
        <div className="source-field">
          <div className="field-heading">
            <label htmlFor="source">Starting text</label>
            <div className="field-tools">
              <button
                className="random-source"
                type="button"
                onClick={pickRandomSource}
                disabled={loading}
                aria-label="Pick random text"
              >
                random ↻
              </button>
              <span>{source.length}/240</span>
            </div>
          </div>
          <textarea
            id="source"
            ref={sourceRef}
            value={source}
            onChange={(event) => setSource(event.target.value)}
            maxLength={MAX_SOURCE_LENGTH}
            rows={4}
            placeholder="Please read the manual."
            disabled={loading}
          />
          <p className="field-note">
            Edit this starting point, or pick another one.
          </p>
        </div>

        <div className="controls-grid">
          <div className="select-field">
            <label htmlFor="dimension">Tone dimension</label>
            <div className="select-wrap">
              <select
                id="dimension"
                value={dimensionKey}
                onChange={(event) =>
                  setDimensionKey(event.target.value as DimensionKey)
                }
                disabled={loading}
              >
                {sortedDimensions.map((key) => (
                  <option key={key} value={key}>
                    {DIMENSIONS[key].name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="dial-field">
            <div className="dial-heading">
              <label htmlFor="target">{dimension.name} level</label>
              <output htmlFor="target">{target}%</output>
            </div>
            <input
              id="target"
              type="range"
              min="0"
              max="100"
              step="5"
              value={target}
              onChange={(event) => setTarget(Number(event.target.value))}
              disabled={loading}
              aria-label={`${dimension.name} level`}
              style={{ "--target": `${target}%` } as CSSProperties}
            />
            <div className="dial-ends" aria-hidden="true">
              <span>{dimension.low}</span>
              <span>{dimension.high}</span>
            </div>
          </div>
        </div>

        <div className="submit-row">
          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "toning…" : "tone it"}
            <span aria-hidden="true"> ←</span>
          </button>
          <p className="process-status" role="status" aria-live="polite">
            {message}
          </p>
        </div>

        {error && (
          <aside className="error-panel" role="alert">
            <strong>{error}</strong>
            {error.includes("Try again soon") && (
              <p>Give the public tone machine a few seconds to cool off.</p>
            )}
          </aside>
        )}

        {loading && streamAttempts.length > 0 && (
          <section className="live-loop" aria-label="Live tone attempts">
            <p className="live-loop-heading">Jev scores so far</p>
            <ol>
              {streamAttempts.map((attempt, index) => (
                <li key={`${index}-${attempt.phrase}`}>
                  <div className="live-attempt-heading">
                    <span>attempt {index + 1}</span>
                    <strong>{scoreText(attempt.score)}%</strong>
                  </div>
                  <div
                    className="live-score"
                    role="progressbar"
                    aria-label={`Attempt ${index + 1}, ${scoreText(attempt.score)}% ${dimension.name}`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={attempt.score}
                  >
                    <span style={{ width: `${attempt.score}%` }} />
                  </div>
                  <p>{attempt.phrase}</p>
                </li>
              ))}
            </ol>
          </section>
        )}
      </form>

      {result && (
        <section className="result-card" aria-label="Toned result">
          <div className="result-heading">
            <div>
              <p className="kicker">best of {result.attempts.length}</p>
              <h2>{result.hit ? "Close enough." : "Closest one."}</h2>
            </div>
            <div className={`verdict ${result.hit ? "is-hit" : "is-miss"}`}>
              {verdictText(result)}
            </div>
          </div>

          <blockquote>{result.phrase}</blockquote>

          <dl className="score-summary">
            <div>
              <dt>asked for</dt>
              <dd>{scoreText(result.target)}%</dd>
            </div>
            <div>
              <dt>Jev says</dt>
              <dd>{scoreText(result.score)}%</dd>
            </div>
            <div>
              <dt>distance</dt>
              <dd>{scoreText(result.distance)} points away</dd>
            </div>
          </dl>

          <div className="result-actions">
            <button type="button" onClick={copyResult}>
              copy result
            </button>
            <button type="button" onClick={reuseResult}>
              use as starting text
            </button>
          </div>

          <details className="attempts">
            <summary>show the loop</summary>
            <ol>
              {result.attempts.map((attempt, index) => (
                <li key={`${index}-${attempt.phrase}`}>
                  <span>
                    attempt {index + 1} · {scoreText(attempt.score)}%
                  </span>
                  <p>“{attempt.phrase}”</p>
                </li>
              ))}
            </ol>
          </details>
        </section>
      )}

      <footer className="site-footer">
        <span>Granite writes · TypeSafe Jev scores · nothing is saved</span>
        <div className="footer-actions">
          <a
            href="#inspect-api"
            className="text-button"
            ref={inspectApiLinkRef}
            onClick={() => setApiInspectorOpen(true)}
          >
            inspect API
          </a>
          <a
            href="https://github.com/tomviner/tone-your-mind"
            target="_blank"
            rel="noreferrer"
          >
            source
          </a>
        </div>
      </footer>

      {apiInspectorOpen && (
        <ApiInspector
          entries={apiLog}
          onClear={() => setApiLog([])}
          onClose={closeApiInspector}
        />
      )}
    </main>
  );
}
