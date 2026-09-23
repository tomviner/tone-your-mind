import type { DimensionKey } from "./dimensions";

export interface ToneAttempt {
  phrase: string;
  score: number;
  confidence: number | null;
}

export interface ToneResponse {
  phrase: string;
  dimension: DimensionKey;
  target: number;
  score: number;
  distance: number;
  hit: boolean;
  attempts: ToneAttempt[];
  models: {
    writer: string;
    scorer: string | null;
  };
  inspection?: {
    model_calls?: unknown[];
  };
}
