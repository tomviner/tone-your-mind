import {
  type CSSProperties,
  type FormEvent,
  useMemo,
  useRef,
  useState,
} from "react";

import { DIMENSIONS, DIMENSION_KEYS, type DimensionKey } from "./dimensions";
import type { ToneResponse } from "./types";

const MAX_SOURCE_LENGTH = 240;
const SESSION_STORAGE_KEY = "tone-jev-session";
const SESSION_PATTERN = /^[A-Za-z0-9_-]{16,64}$/;

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

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isPercentage = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 100;

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
  const [source, setSource] = useState("");
  const [dimensionKey, setDimensionKey] = useState<DimensionKey>("panic");
  const [target, setTarget] = useState(60);
  const [result, setResult] = useState<ToneResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(
    "Set the dial. Granite writes; Jev judges.",
  );
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<HTMLTextAreaElement>(null);
  const dimension = DIMENSIONS[dimensionKey];
  const sortedDimensions = useMemo(
    () =>
      [...DIMENSION_KEYS].sort((left, right) =>
        DIMENSIONS[left].name.localeCompare(DIMENSIONS[right].name),
      ),
    [],
  );

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (loading) return;

    setLoading(true);
    setError(null);
    setMessage("Granite writes. Jev judges. The loop tightens.");

    try {
      const response = await fetch("/api/tone", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-tone-session": sessionId(),
        },
        body: JSON.stringify({ source, dimension: dimensionKey, target }),
      });
      const body: unknown = await response.json();
      if (!response.ok) {
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

  return (
    <main className="tool-shell">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="tone JEV home">
          tone <span>JEV</span>
        </a>
        <a className="sibling-link" href="https://jev-tone.tomv.uk">
          play the original <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero" aria-labelledby="page-title">
        <p className="kicker">the machine plays the game now</p>
        <h1 id="page-title">
          <span>tone</span> <strong>JEV</strong>
        </h1>
        <p className="strapline">tone your mind</p>
        <p className="lede">
          Need to panic someone, but only 60%? Pick a feeling, turn the dial,
          and let one AI rewrite while another marks its homework.
        </p>
      </section>

      <form className="tone-card" onSubmit={submit} aria-busy={loading}>
        <div className="source-field">
          <div className="field-heading">
            <label htmlFor="source">Starting text</label>
            <span>{source.length}/240</span>
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
            Optional—leave it blank and the machine invents something to tone.
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
      </form>

      {result && (
        <section className="result-card" aria-label="Toned result">
          <div className="result-heading">
            <div>
              <p className="kicker">best of {result.attempts.length}</p>
              <h2>{result.hit ? "Close enough." : "Closest one."}</h2>
            </div>
            <div className={`verdict ${result.hit ? "is-hit" : "is-miss"}`}>
              {result.hit ? "hit" : "near miss"}
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

      <aside className="reality-check">
        <strong>Check before sending.</strong> The writer can change meaning as
        well as tone. Jev measures a rubric; it does not know your relationship.
      </aside>

      <footer className="site-footer">
        <span>Granite writes · TypeSafe Jev scores · nothing is saved</span>
        <a
          href="https://github.com/tomviner/tone-your-mind"
          target="_blank"
          rel="noreferrer"
        >
          GitHub repo <span aria-hidden="true">↗</span>
        </a>
      </footer>
    </main>
  );
}
