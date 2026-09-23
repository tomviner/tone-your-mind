export interface ApiLogEntry {
  id: number;
  request: unknown;
  response?: unknown;
  modelCalls?: unknown[];
  httpStatus?: number;
  status: "pending" | "complete" | "error";
}

interface ApiInspectorProps {
  entries: ApiLogEntry[];
  onClear: () => void;
  onClose: () => void;
}

const json = (value: unknown) => JSON.stringify(value, null, 2);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const callLabel = (value: unknown): string =>
  isRecord(value) && value.kind === "writer" ? "Granite" : "Jev";

const callField = (value: unknown, key: "request" | "response"): unknown =>
  isRecord(value) ? value[key] : undefined;

export default function ApiInspector({
  entries,
  onClear,
  onClose,
}: ApiInspectorProps) {
  return (
    <section
      className="api-inspector"
      id="inspect-api"
      aria-label="API inspector"
    >
      <div className="api-inspector-heading">
        <div>
          <p className="kicker">under the hood</p>
          <h2>Inspect API</h2>
        </div>
        <div className="api-inspector-actions">
          {entries.length > 0 && (
            <button className="text-button" type="button" onClick={onClear}>
              clear log
            </button>
          )}
          <button className="text-button" type="button" onClick={onClose}>
            close
          </button>
        </div>
      </div>
      <p className="api-inspector-note">
        Browser session only. Exact API and model inputs; normalized model
        responses. Nothing here is saved as history.
      </p>

      {entries.length === 0 ? (
        <div className="api-empty">
          <strong>POST /api/tone</strong>
          <p>Tone a message and its live exchanges will appear here.</p>
          <pre>
            <code>
              {json({
                source: "Please read the manual.",
                dimension: "panic",
                target: 60,
              })}
            </code>
          </pre>
        </div>
      ) : (
        <div className="api-log">
          {[...entries].reverse().map((entry, index) => (
            <details className="api-entry" key={entry.id} open={index === 0}>
              <summary>
                <span>POST /api/tone</span>
                <span className={`api-status is-${entry.status}`}>
                  {entry.httpStatus ?? entry.status}
                </span>
              </summary>
              <div className="api-entry-body">
                <h3>Browser request</h3>
                <pre>
                  <code>{json(entry.request)}</code>
                </pre>

                {entry.modelCalls?.map((call, callIndex) => {
                  const label = callLabel(call);
                  return (
                    <div className="model-call" key={callIndex}>
                      <p className="model-call-number">
                        model call {callIndex + 1}
                      </p>
                      <h3>{label} request</h3>
                      <pre>
                        <code>{json(callField(call, "request"))}</code>
                      </pre>
                      <h3>{label} response</h3>
                      <pre>
                        <code>
                          {callField(call, "response") === null
                            ? "No model response."
                            : json(callField(call, "response"))}
                        </code>
                      </pre>
                    </div>
                  );
                })}

                <h3>API response</h3>
                <pre>
                  <code>
                    {entry.response === undefined
                      ? entry.status === "pending"
                        ? "Waiting for Granite and Jev…"
                        : "No response."
                      : json(entry.response)}
                  </code>
                </pre>
              </div>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}
