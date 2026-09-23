import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";

const SESSION = "123e4567-e89b-12d3-a456-426614174000";

const resultBody = {
  phrase: "Please read the manual right now—something is very wrong.",
  dimension: "panic",
  target: 60,
  score: 62.5,
  distance: 2.5,
  hit: true,
  attempts: [
    {
      phrase: "Please read the manual.",
      score: 12.5,
      confidence: 0.91,
    },
    {
      phrase: "Please read the manual right now—something is very wrong.",
      score: 62.5,
      confidence: 0.88,
    },
  ],
  models: {
    writer: "@cf/ibm-granite/granite-4.0-h-micro",
    scorer: "jev-1.13.0",
  },
  inspection: {
    model_calls: [
      {
        kind: "writer",
        request: {
          model: "@cf/ibm-granite/granite-4.0-h-micro",
          input: { messages: [{ role: "user", content: "Rewrite this." }] },
        },
        response: { phrase: "Please read the manual right now." },
      },
      {
        kind: "scorer",
        request: {
          model: "typesafe/jev",
          input: { state: "Please read the manual right now." },
        },
        response: {
          model: "jev-1.13.0",
          score: 2.5,
          confidence: 0.88,
          reviewer_feedback: {
            legend: {
              0: "Unruffled",
              1: "Slight concern",
              2: "Clearly worried",
              3: "Strong panic",
              4: "Full panic",
            },
            probabilities: { 0: 0, 1: 0, 2: 0.5, 3: 0.5, 4: 0 },
          },
        },
      },
    ],
  },
};

const prepareSession = () => {
  window.localStorage.setItem("tone-jev-session", SESSION);
};

const typeSource = (value = "Please read the manual.") => {
  fireEvent.change(screen.getByLabelText(/starting text/i), {
    target: { value },
  });
};

const submit = () => {
  fireEvent.click(screen.getByRole("button", { name: /tone it/i }));
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.localStorage.clear();
  window.history.replaceState(null, "", "/");
});

describe("tone your mind", () => {
  test("publishes the real DSPy prompt lab at /lab", () => {
    window.history.replaceState(null, "", "/lab");
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: /Teaching a tiny model to hit the dial/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("DSPy MIPROv2", { exact: false }),
    ).toBeInTheDocument();
    expect(screen.getByText(/85%/)).toBeInTheDocument();
    expect(screen.getByText(/FINAL ATTEMPT/)).toBeInTheDocument();
    expect(
      screen.getByText(/What Granite is actually told/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/59.8 → 61.2/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "source" })).toHaveAttribute(
      "href",
      "https://github.com/tomviner/tone-your-mind",
    );
  });

  test("hands the current starting text back to the original game", () => {
    render(<App />);

    const link = within(screen.getByRole("banner")).getByRole("link", {
      name: "mind your tone",
    });
    const url = new URL(link.getAttribute("href")!);

    expect(link).toHaveTextContent("← mind your tone");
    expect(url.origin).toBe("https://jev-tone.tomv.uk");
    expect(url.searchParams.get("text")).toBe("Please read the manual.");
  });

  test("loads a shared tone recipe from the URL", () => {
    window.history.replaceState(
      null,
      "",
      "/?text=Could%20you%20send%20that%3F&dimension=whimsy&target=95",
    );
    render(<App />);

    expect(screen.getByLabelText(/starting text/i)).toHaveValue(
      "Could you send that?",
    );
    expect(screen.getByLabelText(/tone dimension/i)).toHaveValue("whimsy");
    expect(screen.getByRole("slider", { name: /whimsy level/i })).toHaveValue(
      "95",
    );
  });

  test("sanitizes malformed tone recipes at the URL boundary", () => {
    const supplied = "x".repeat(260);
    window.history.replaceState(
      null,
      "",
      `/?text=${encodeURIComponent(supplied)}&dimension=unknown&target=104`,
    );

    render(<App />);

    expect(screen.getByLabelText(/starting text/i)).toHaveValue(
      "x".repeat(240),
    );
    expect(screen.getByLabelText(/tone dimension/i)).toHaveValue("panic");
    expect(screen.getByRole("slider", { name: /panic level/i })).toHaveValue(
      "60",
    );
  });

  test("keeps the editable tone recipe shareable", () => {
    render(<App />);

    typeSource("A shared line.");
    fireEvent.change(screen.getByLabelText(/tone dimension/i), {
      target: { value: "whimsy" },
    });
    fireEvent.change(screen.getByRole("slider", { name: /whimsy level/i }), {
      target: { value: "95" },
    });

    const link = screen.getByRole("link", { name: "share recipe" });
    const url = new URL(link.getAttribute("href")!, window.location.href);

    expect(url.searchParams.get("text")).toBe("A shared line.");
    expect(url.searchParams.get("dimension")).toBe("whimsy");
    expect(url.searchParams.get("target")).toBe("95");
    expect(url.hash).toBe("");
  });

  test("links the writer and scorer credits to their model documentation", () => {
    render(<App />);

    const header = within(screen.getByRole("banner"));
    const footer = within(screen.getByRole("contentinfo"));
    const graniteLinks = [
      header.getByRole("link", { name: "Granite" }),
      footer.getByRole("link", { name: "Granite" }),
    ];
    const jevLinks = [
      header.getByRole("link", { name: "Jev" }),
      footer.getByRole("link", { name: "Jev" }),
    ];

    for (const link of graniteLinks) {
      expect(link).toHaveAttribute(
        "href",
        "https://developers.cloudflare.com/workers-ai/models/granite-4.0-h-micro/",
      );
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noreferrer");
    }
    for (const link of jevLinks) {
      expect(link).toHaveAttribute(
        "href",
        "https://developers.cloudflare.com/ai/models/typesafe/jev/",
      );
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noreferrer");
    }
  });

  test("leads with the inverse action instead of the internal tone JEV name", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Pick a tone. Let the machine write.",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "tone your mind home" }),
    ).toHaveTextContent("tone your mind.");
    expect(screen.getByLabelText(/tone dimension/i)).toHaveValue("panic");
    expect(screen.getByRole("slider", { name: /panic level/i })).toHaveValue(
      "60",
    );
    expect(screen.getByText("Unruffled")).toBeInTheDocument();
    expect(screen.getByText("Full panic")).toBeInTheDocument();
    expect(screen.getByLabelText(/starting text/i)).toHaveAttribute(
      "maxlength",
      "240",
    );
    expect(screen.getByLabelText(/starting text/i)).toHaveValue(
      "Please read the manual.",
    );
    expect(
      screen.getByRole("button", { name: /random text/i }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/check before sending/i)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/doesn’t guarantee the meaning is maintained/i),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /^source/i })).toHaveAttribute(
      "href",
      "https://github.com/tomviner/tone-your-mind",
    );
    expect(
      screen.queryByRole("link", { name: /github repo/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toHaveTextContent(
      "Granite writes · TypeSafe Jev scores · nothing is saved",
    );
  });

  test("picks a different visible starting text", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /random text/i }));

    expect(screen.getByLabelText(/starting text/i)).not.toHaveValue(
      "Please read the manual.",
    );
  });

  test("updates the dial labels when a different dimension is chosen", () => {
    render(<App />);

    fireEvent.change(screen.getByLabelText(/tone dimension/i), {
      target: { value: "smugness" },
    });
    fireEvent.change(screen.getByRole("slider", { name: /smugness level/i }), {
      target: { value: "75" },
    });

    expect(screen.getByText("Self-effacing")).toBeInTheDocument();
    expect(screen.getByText("Insufferably smug")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  test("submits optional source, target, dimension, and browser session", async () => {
    prepareSession();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        Response.json(resultBody),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    typeSource();

    submit();

    expect(await screen.findByText(resultBody.phrase)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/tone");
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({
      accept: "application/x-ndjson",
      "content-type": "application/json",
      "x-tone-session": SESSION,
    });
    expect(JSON.parse(String(init?.body))).toEqual({
      source: "Please read the manual.",
      dimension: "panic",
      target: 60,
    });
  });

  test("hands the best machine attempt back to the original game", async () => {
    prepareSession();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json(resultBody)),
    );
    render(<App />);

    submit();
    await screen.findByText(resultBody.phrase);

    const link = within(screen.getByRole("banner")).getByRole("link", {
      name: "mind your tone",
    });
    const url = new URL(link.getAttribute("href")!);
    expect(url.searchParams.get("text")).toBe(resultBody.phrase);
  });

  test("shows measured truth and the complete attempt trail", async () => {
    prepareSession();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json(resultBody)),
    );
    render(<App />);
    typeSource();

    submit();

    const result = await screen.findByRole("region", { name: /toned result/i });
    expect(within(result).getByText("asked for")).toBeInTheDocument();
    expect(within(result).getByText("60%")).toBeInTheDocument();
    expect(within(result).getByText("Jev says")).toBeInTheDocument();
    expect(within(result).getByText("62.5%")).toBeInTheDocument();
    expect(within(result).getByText("2.5 points away")).toBeInTheDocument();
    expect(within(result).getByText("Close enough.")).toBeInTheDocument();
    expect(within(result).getByText(/machine attempt/i)).toBeInTheDocument();
    expect(within(result).getAllByRole("listitem")).toHaveLength(2);
    expect(within(result).getByText("attempt 1 · 12.5%")).toBeInTheDocument();
    expect(within(result).getByText("attempt 2 · 62.5%")).toBeInTheDocument();
  });

  test("grades misses by how far Jev landed from the target", async () => {
    const cases = [
      { distance: 12, score: 68, label: "near miss" },
      { distance: 30, score: 50, label: "not quite" },
      { distance: 80, score: 0, label: "way off" },
    ] as const;

    for (const example of cases) {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () =>
          Response.json({
            ...resultBody,
            phrase: `Result ${example.distance} points away.`,
            target: 80,
            score: example.score,
            distance: example.distance,
            hit: false,
          }),
        ),
      );
      const view = render(<App />);

      submit();

      const result = await screen.findByRole("region", {
        name: /toned result/i,
      });
      expect(within(result).getByText(example.label)).toBeInTheDocument();
      view.unmount();
      vi.unstubAllGlobals();
    }
  });

  test("opens an inspector showing the real API, Granite, and Jev exchanges", async () => {
    prepareSession();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json(resultBody)),
    );
    render(<App />);
    typeSource();
    submit();
    await screen.findByText(resultBody.phrase);

    fireEvent.click(screen.getByRole("link", { name: /inspect api/i }));

    const inspector = screen.getByRole("region", { name: /api inspector/i });
    expect(within(inspector).getByText("POST /api/tone")).toBeInTheDocument();
    expect(within(inspector).getByText("Browser request")).toBeInTheDocument();
    expect(within(inspector).getByText("Granite request")).toBeInTheDocument();
    expect(within(inspector).getByText("Granite response")).toBeInTheDocument();
    expect(within(inspector).getByText("Jev request")).toBeInTheDocument();
    expect(
      within(inspector).getByText("Reviewer feedback · Jev response"),
    ).toBeInTheDocument();
    expect(inspector).toHaveTextContent('"4": "Full panic"');
    expect(inspector).toHaveTextContent('"3": 0.5');
    expect(within(inspector).getByText("API response")).toBeInTheDocument();
    expect(within(inspector).getAllByText(/typesafe\/jev/)).not.toHaveLength(0);
    expect(
      within(inspector).getAllByText(/granite-4\.0-h-micro/),
    ).not.toHaveLength(0);
  });

  test("shows each streamed score before the final result arrives", async () => {
    prepareSession();
    const encoder = new TextEncoder();
    let finishStream: (() => void) | undefined;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            `${JSON.stringify({
              type: "attempt",
              attempt: {
                phrase: "Please read the manual.",
                score: 12.5,
                confidence: 0.91,
              },
            })}\n`,
          ),
        );
        finishStream = () => {
          controller.enqueue(
            encoder.encode(
              `${JSON.stringify({ type: "complete", status: 200, result: resultBody })}\n`,
            ),
          );
          controller.close();
        };
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Promise.resolve(
          new Response(stream, {
            headers: { "content-type": "application/x-ndjson" },
          }),
        ),
      ),
    );
    render(<App />);

    submit();

    const progress = await screen.findByRole("progressbar", {
      name: /attempt 1.*12.5% panic/i,
    });
    expect(progress).toHaveAttribute("aria-valuenow", "12.5");
    expect(screen.queryByRole("region", { name: /toned result/i })).toBeNull();

    finishStream?.();
    expect(
      await screen.findByRole("region", { name: /toned result/i }),
    ).toBeInTheDocument();
  });

  test("copies and reuses the best result", async () => {
    prepareSession();
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json(resultBody)),
    );
    render(<App />);
    submit();
    await screen.findByText(resultBody.phrase);

    fireEvent.click(screen.getByRole("button", { name: /copy result/i }));
    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(resultBody.phrase),
    );
    expect(screen.getByText("Copied.")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /use as starting text/i }),
    );
    expect(screen.getByLabelText(/starting text/i)).toHaveValue(
      resultBody.phrase,
    );
    expect(screen.getByLabelText(/starting text/i)).toHaveFocus();
  });

  test("allows only one in-flight request", () => {
    prepareSession();
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise<Response>(() => undefined)),
    );
    render(<App />);

    submit();
    const button = screen.getByRole("button", { name: /toning/i });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(fetch).toHaveBeenCalledTimes(1);
    const status = screen.getByText("Jev is scoring…");
    expect(status).toHaveClass("process-status");
    expect(status).toHaveTextContent("Jev is scoring…");
    expect(status.querySelector(".status-spinner")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
  });

  test("recovers from model errors without replacing an existing result", async () => {
    prepareSession();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json(resultBody))
      .mockResolvedValueOnce(
        Response.json(
          { error: "The tone loop lost the plot" },
          { status: 502 },
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    submit();
    await screen.findByText(resultBody.phrase);

    submit();

    expect(await screen.findByText(/lost the plot/i)).toBeInTheDocument();
    expect(screen.getByText(resultBody.phrase)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /tone it/i })).not.toBeDisabled();
  });

  test("treats a malformed successful response as a recoverable error", async () => {
    prepareSession();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ phrase: "missing everything else" })),
    );
    render(<App />);

    submit();

    expect(
      await screen.findByText(/tone loop returned an invalid result/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: /toned result/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /tone it/i })).not.toBeDisabled();
  });

  test("explains rate limiting and makes retry possible", async () => {
    prepareSession();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json(
          { error: "Easy, tiger. Try again soon." },
          { status: 429 },
        ),
      ),
    );
    render(<App />);

    submit();

    expect(await screen.findByText(/Easy, tiger/i)).toBeInTheDocument();
    expect(screen.getByText(/few seconds/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /tone it/i })).not.toBeDisabled();
  });
});
