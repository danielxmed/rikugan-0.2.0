export type ParsedCommand =
  | { kind: 'load'; modelName: string }
  | { kind: 'run'; prompt: string }
  | { kind: 'info' }
  | { kind: 'help'; topic: string | null }
  | { kind: 'contrast'; gamma: number }
  | { kind: 'slice'; depth: number | null }
  | { kind: 'error'; message: string }
  | { kind: 'unknown'; raw: string }
  | { kind: 'empty' };
