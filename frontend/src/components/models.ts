export interface Todo {
  id: number;
  content: string;
}

export interface GenerationParams {
  doSample: boolean;
  minNewTokens: number;
  maxNewTokens: number;
  repetitionPenalty: number;
  temperature: number;
  topP: number;
  topK: number;
}

export interface Meta {
  totalCount: number;
}
