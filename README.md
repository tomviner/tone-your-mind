# tone JEV

`tone your mind`: the playful inverse of
[mind your tone](https://jev-tone.tomv.uk).

**Try it:** [tone-jev.tomv.uk](https://tone-jev.tomv.uk)

**See the prompt lab:** [tone-jev.tomv.uk/lab](https://tone-jev.tomv.uk/lab)

Edit the visible starting text (or pick a random one), choose a tone dimension,
and turn a 0–100 dial. An inexpensive language model rewrites the text while
TypeSafe's Jev model scores each attempt. Scored attempts stream into the page
as they arrive; the closest version wins.

Panic starts at 60%, because the whole project began with a question:

> Can this be inverted? Like I need to panic someone but only 60%?

## The loop

1. If starting text exists, Jev scores it first.
2. IBM Granite 4.0 H Micro receives the source, the target, and every previous
   phrase with its real Jev score.
3. Granite returns one short rewrite; Jev scores it on the selected five-anchor
   rubric.
4. The loop stops within five percentage points or after four total scores.
5. The closest attempt is returned, even when none hits the tolerance.

Writer responses that mention scoring metadata, or introduce numerals absent
from the starting text, are rejected before they reach Jev or the browser. A
rejection still consumes one of the bounded writer calls.

The deployed instruction is compiled in the public DSPy prompt lab. Tone
accuracy is weighted at 85%; Jev's probability that the core proposition
survived contributes the remaining 15%. This makes hitting the dial the primary
objective while still discouraging gratuitous semantic drift. Jev measures the
wording, but it does not guarantee the meaning is maintained.

After each miss, one of five distance bands tells Granite how hard to correct.
The retry number adds a second multiplier: the first correction is firm, the
second emphatic, and the final one deliberately melodramatic, superlative, and
insistent. These are private instructions; output that mentions scores or the
editing machinery is rejected.

A source-based run therefore makes at most three writer calls and four Jev
calls. A blank-source run makes at most four of each. Writer output is capped at
96 tokens and every phrase at 240 characters.

The writer is
[`@cf/ibm-granite/granite-4.0-h-micro`](https://developers.cloudflare.com/workers-ai/models/granite-4.0-h-micro/),
Cloudflare's least expensive listed text-generation model when this project was
built. Jev remains the only scoring authority.

## Architecture

- React and Vite render a mobile-first single-dial interface.
- `POST /api/tone` is a same-origin Python Cloudflare Worker endpoint.
- Clients receive ordinary JSON by default. The browser requests
  `application/x-ndjson`, so the Worker streams one scored-attempt event at a
  time followed by the same final result in a completion event.
- The Worker uses the native Workers AI binding for both Granite and
  `typesafe/jev`; there are no API keys in application code.
- Native Cloudflare bindings allow six requests per browser session and thirty
  aggregate requests per edge location each minute.
- The browser stores only an opaque rate-limit session ID. The server stores no
  phrases, generations, accounts, history, or analytics.

The response includes the winning phrase, requested and measured percentages,
the complete attempt trail, and an inspection trace. The expandable **inspect
API** section shows the exact browser request, the ordered Granite and Jev model
inputs, normalized model responses, Jev's full per-level probability feedback,
and the complete API response. Its log lives only in the current browser
session. The interface is deliberately honest about near misses.

## Prompt lab

[`/lab`](https://tone-jev.tomv.uk/lab) publishes the selected system
instruction, all five live feedback bands, retry escalation, fixed held-out
examples, and the actual optimisation outcome—including misses. Production
imports `worker/prompt_program.json`, the same versioned artifact shown by the
lab; it does not maintain a hidden second prompt.

The offline compiler uses DSPy MIPROv2 with Granite as both task and prompt
model. Jev reviews each candidate with two questions in one call: the requested
tone as a five-anchor score, and meaning retention as a Noul probability. Large
tone misses are squared so one severe failure matters more than several small
gains. Only `lab/dataset.json` is used; user text and production logs are never
training data.

To reproduce and publish a five-trial run:

```bash
export CLOUDFLARE_ACCOUNT_ID=…
export CLOUDFLARE_API_TOKEN=…
# Optional; defaults to the account's `default` AI Gateway.
export CLOUDFLARE_AI_GATEWAY_ID=…
uv run --group lab python -m lab.optimize --trials 5 --publish
```

## Development

Requires Node.js 20.19 or newer, Python 3.13 or newer, and
[`uv`](https://docs.astral.sh/uv/).

```bash
npm ci
uv sync --locked
npm run format:check
npm run lint:py
npm test
npm run typecheck
npm run build
uv run pywrangler dev
```

Workers AI runs remotely. The AI, static assets, and two rate-limit bindings are
declared in `wrangler.jsonc`.

Production deployment is intentional and manual:

```bash
npm run build
uv run pywrangler deploy
```

The Python Worker and React assets deploy as one unit, and the custom domain
configuration owns `tone-jev.tomv.uk`. A manual GitHub Actions deployment is
also included for use once repository Cloudflare credentials are configured.

## License

MIT
