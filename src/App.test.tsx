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
});

describe("tone JEV", () => {
  test("opens as the inverse tool with Panic at 60 percent", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "tone JEV" }),
    ).toBeInTheDocument();
    expect(screen.getByText("tone your mind")).toBeInTheDocument();
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
    expect(screen.getByText(/leave it blank/i)).toBeInTheDocument();
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
      "content-type": "application/json",
      "x-tone-session": SESSION,
    });
    expect(JSON.parse(String(init?.body))).toEqual({
      source: "Please read the manual.",
      dimension: "panic",
      target: 60,
    });
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
    expect(within(result).getAllByRole("listitem")).toHaveLength(2);
    expect(within(result).getByText("attempt 1 · 12.5%")).toBeInTheDocument();
    expect(within(result).getByText("attempt 2 · 62.5%")).toBeInTheDocument();
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
    expect(screen.getByText(/Granite writes.*Jev judges/i)).toBeInTheDocument();
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
