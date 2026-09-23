# tone JEV — Design

## Intent

Build the playful inverse of `mind your tone`: instead of a person editing a
phrase until Jev accepts its tone, a small language model edits the phrase and
Jev judges each attempt. The tool should answer the joke, “I need to panic
someone, but only 60%,” quickly enough to invite experimentation and cheaply
enough to share publicly.

Success means a visitor can provide optional source text, choose one tone,
dial a percentage, and receive a short usable rewrite whose measured Jev score
is visibly close to the requested value. A miss must be presented honestly.

## Product

The visible title is `tone JEV`; the strapline is `tone your mind`. The initial
control is Panic at 60%, matching the conversation that inspired the project.
The source textarea is optional and limited to 240 characters. When omitted,
the writer invents a short everyday message rather than asking another question.

One dimension is active at a time. The picker contains every dimension from
`mind your tone`, plus Panic, Passive aggression, Smugness, Drama, and
Ominousness. The slider runs from 0% to 100% in 5-point steps and labels both
ends with the dimension's Jev anchors.

Pressing `tone it` replaces the result only after the whole operation succeeds.
The result shows the best phrase, requested percentage, measured percentage,
distance from target, and every attempt in order. A copy button copies the best
phrase. The source can be replaced with the result for another pass.

## Feedback loop

Jev remains the only scoring authority. A target percentage maps linearly to
Jev's 0–4 Score scale. If source text exists, it is scored first as attempt zero.
The writer then receives the source, selected rubric, requested score, and the
complete bounded history of phrases and Jev scores. It returns only a revised
phrase. Jev scores that phrase, and the history is fed into the next iteration.

The loop stops when an attempt is within 5 percentage points or when four total
scored attempts have been made. The closest attempt wins, with earlier attempts
winning ties. A blank-source run generates and scores up to four candidates; a
source-based run scores the source and generates at most three rewrites.

The writer is `@cf/ibm-granite/granite-4.0-h-micro`, selected because Cloudflare
lists it as its least expensive text-generation model and describes it as an
instruction-following edge model. Outputs are capped at 96 tokens. Prompts and
history are bounded by the 240-character phrase limit and four attempts.

## Architecture

The project reuses the React/Vite and Python Cloudflare Worker structure of
`mind your tone`. Shared JSON describes each tone and its five Jev anchors. Pure
Python functions validate requests, construct writer prompts and Jev inputs,
parse both model response shapes, and run the feedback loop through an injected
AI runner. The HTTP adapter exposes only `POST /api/tone`; static assets are
served by the Worker.

Two native Cloudflare rate-limit bindings protect inference: six requests per
minute per browser session and thirty total requests per minute per Cloudflare
location. The browser creates an opaque session identifier in local storage and
sends it in `X-Tone-Session`; it contains no user identity. Requests without a
valid identifier are rejected before inference.

## Interface

The sibling relationship is obvious but inverted: cool dark ink, vivid coral,
mint and warm cream replace the original yellow/blue paper palette; the title is
typographically reversed; the score rail is controlled by the person rather
than moved by Jev. The page is mobile-first, keyboard usable, explicit about its
busy state, and respects reduced motion.

The browser displays fixed status copy while the feedback loop runs. It does not
stream intermediate prose, because showing a half-complete attempt complicates
retries and encourages duplicate submissions. A request can be cancelled in
the browser, while the server's strict loop bound contains work already begun.

## Errors, privacy, and cost

Malformed JSON, cross-origin requests, missing session IDs, unknown dimensions,
non-step targets, and overlong text fail before inference. Rate-limited requests
return 429. Writer or Jev failures return a generic retryable error and no
partial generated text. Responses use `no-store`; source text, generations, and
history are never persisted or deliberately logged.

The UI states that AI may alter meaning and asks users to check before sending.
No accounts, database, analytics, server-side history, or user-supplied rubric
text are included. Those omissions reduce privacy risk, prompt size, and abuse
surface.

## Verification

Python tests cover request validation, prompt construction, response parsing,
source and blank-source loops, early success, attempt caps, best-attempt choice,
and upstream failures. React tests cover defaults, slider and dimension changes,
submission, results, retry, copying, and rate-limit copy. Full formatting,
linting, tests, typechecking, and production build run before deployment. The
deployed custom domain and API method handling are verified directly.

