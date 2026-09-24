import results from "./lab-results.json";
import promptProgram from "../worker/prompt_program.json";

const formatPoints = (value: number) => `${value.toFixed(1)}%`;

export default function Lab() {
  return (
    <main className="tool-shell lab-shell">
      <header className="site-header lab-header">
        <a className="wordmark" href="/" aria-label="tone your mind home">
          tone your mind<span>.</span>
        </a>
        <a className="sibling-link" href="/">
          use the tool <span aria-hidden="true">→</span>
        </a>
      </header>

      <section className="lab-hero" aria-labelledby="lab-title">
        <p className="kicker">public prompt lab · {promptProgram.version}</p>
        <h1 id="lab-title">Teaching a tiny model to hit the dial.</h1>
        <p className="lede">
          Granite writes. Jev judges. DSPy changes the instruction, runs the
          fixed examples again, and keeps the version that scores better.
          Production uses the prompt published on this page.
        </p>
      </section>

      <section className="lab-section" aria-labelledby="two-loops-title">
        <p className="kicker">the mechanism</p>
        <h2 id="two-loops-title">Two feedback loops, different speeds.</h2>
        <div className="loop-grid">
          <article>
            <span className="loop-number">01</span>
            <h3>Live loop</h3>
            <p>
              Your text is scored, Granite rewrites it, and Jev scores the
              result. That repeats for at most four scores. Only the closest
              attempt survives.
            </p>
          </article>
          <article>
            <span className="loop-number">02</span>
            <h3>Lab loop</h3>
            <p>
              DSPy MIPROv2 rewrites the wider writing prompt Granite receives.
              Fixed examples run through Granite and Jev; the stronger prompt
              becomes the next versioned production program.
            </p>
          </article>
        </div>
      </section>

      <section className="lab-section" aria-labelledby="metric-title">
        <p className="kicker">the objective</p>
        <h2 id="metric-title">Accuracy first. Meaning still counts.</h2>
        <div className="metric-grid" aria-label="Optimisation metrics">
          <article className="metric-card">
            <span>tone weight</span>
            <strong>{results.weights.tone}%</strong>
            <p>Squared distance from the requested Jev score.</p>
          </article>
          <article className="metric-card">
            <span>meaning weight</span>
            <strong>{results.weights.meaning}%</strong>
            <p>Jev’s probability that the core proposition survived.</p>
          </article>
          <article className="metric-card metric-card-accent">
            <span>balanced eval</span>
            <strong>
              {results.baseline_metric.toFixed(1)} →{" "}
              {results.selected_metric.toFixed(1)}
            </strong>
            <p>Baseline to selected candidate on the same held-out set.</p>
          </article>
        </div>
        <p className="lab-note">
          Large misses are squared, so a catastrophic result cannot hide as
          easily inside a pleasant average. The dataset contains both low and
          high targets. No production text is used.
        </p>
      </section>

      <section className="lab-section" aria-labelledby="rounds-title">
        <p className="kicker">what the lab learned</p>
        <h2 id="rounds-title">A higher number can still lose.</h2>
        <ol className="round-list">
          {results.rounds.map((round, index) => (
            <li key={round.label}>
              <div>
                <strong>{round.label}</strong>
                <span>
                  {round.metric.toFixed(1)} ·{" "}
                  {index === results.rounds.length - 1 ? "current" : "previous"}
                </span>
              </div>
              <p>{round.outcome}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="lab-section" aria-labelledby="feedback-title">
        <p className="kicker">the live correction</p>
        <h2 id="feedback-title">Five gears. Then increasing desperation.</h2>
        <div className="band-list">
          {promptProgram.distance_bands.map((band, index) => (
            <article key={band.max_points}>
              <span>{index + 1}</span>
              <div>
                <h3>Up to {band.max_points} points away</h3>
                <p>{band.instruction}</p>
              </div>
            </article>
          ))}
        </div>
        <div className="escalator-list">
          {promptProgram.attempt_escalators.map((instruction, index) => (
            <p key={instruction}>
              <strong>retry {index + 1}</strong> {instruction}
            </p>
          ))}
        </div>
        <p className="lab-note">
          These retries are the separate live loop. They correct one message
          using the wider prompt DSPy has already selected; they do not rerun
          DSPy.
        </p>
      </section>

      <section className="lab-section" aria-labelledby="prompt-title">
        <p className="kicker">deployed instruction</p>
        <h2 id="prompt-title">What Granite is actually told.</h2>
        <blockquote className="prompt-quote">
          {promptProgram.system_instruction}
        </blockquote>
      </section>

      <section className="lab-section" aria-labelledby="examples-title">
        <p className="kicker">held-out examples</p>
        <h2 id="examples-title">The good, the bad, and the measurable.</h2>
        <div className="example-table-wrap">
          <table className="example-table">
            <thead>
              <tr>
                <th>Request</th>
                <th>Before DSPy</th>
                <th>After DSPy</th>
                <th>Jev target</th>
              </tr>
            </thead>
            <tbody>
              {results.examples.map((example) => (
                <tr key={example.id}>
                  <td>
                    <strong>{example.dimension}</strong>
                    <span>“{example.source}”</span>
                  </td>
                  <td>
                    <span className="comparison-output">
                      {example.before.output}
                    </span>
                    <span className="comparison-score">
                      <span aria-hidden="true">Jev </span>
                      <span
                        className="jev-number"
                        aria-label={`Jev score ${formatPoints(example.before.score)}`}
                      >
                        {formatPoints(example.before.score)}
                      </span>{" "}
                      ·{" "}
                      <span>
                        meaning {formatPoints(example.before.meaning)}
                      </span>
                    </span>
                  </td>
                  <td>
                    <span className="comparison-output">
                      {example.after.output}
                    </span>
                    <span className="comparison-score">
                      <span aria-hidden="true">Jev </span>
                      <span
                        className="jev-number"
                        aria-label={`Jev score ${formatPoints(example.after.score)}`}
                      >
                        {formatPoints(example.after.score)}
                      </span>{" "}
                      ·{" "}
                      <span>meaning {formatPoints(example.after.meaning)}</span>
                    </span>
                  </td>
                  <td>
                    <span
                      className="jev-number"
                      aria-label={`Jev target ${formatPoints(example.target)}`}
                    >
                      {formatPoints(example.target)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="lab-note">{results.method_note}</p>
      </section>

      <section
        className="lab-section lab-reproduce"
        aria-labelledby="reproduce-title"
      >
        <p className="kicker">run it yourself</p>
        <h2 id="reproduce-title">The experiment is in the repo.</h2>
        <pre>
          <code>
            uv run --group lab python -m lab.optimize --trials 5 --publish
          </code>
        </pre>
        <p>
          The command needs a Cloudflare account ID and API token with Workers
          AI access. It reads <code>lab/dataset.json</code>, never user traffic.
        </p>
      </section>

      <footer className="site-footer lab-footer">
        <span>
          fixed examples · no user text · results published warts and all
        </span>
        <div className="footer-actions">
          <a href="https://dspy.ai/3.2.0/learn/optimization/overview/">DSPy</a>
          <a href="https://developers.cloudflare.com/ai/models/typesafe/jev/">
            Jev
          </a>
          <a href="https://github.com/tomviner/tone-your-mind">source</a>
        </div>
      </footer>
    </main>
  );
}
