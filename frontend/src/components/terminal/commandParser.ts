import type { ParsedCommand } from '../../types/commands';

// Named slice -> depth mapping
const SLICE_DEPTH_MAP: Record<string, number | null> = {
  auto: null,
  resid_pre: 0.0,
  attn: 0.3,
  attn_out: 0.3,
  delta_attn: 0.47,
  mlp: 0.7,
  mlp_out: 0.7,
  delta_mlp: 0.92,
  resid_post: 1.0,
};

function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let inQuote: string | null = null;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (inQuote) {
      if (ch === inQuote) {
        inQuote = null;
      } else {
        current += ch;
      }
    } else if (ch === '"' || ch === "'") {
      inQuote = ch;
    } else if (ch === ' ' || ch === '\t') {
      if (current) {
        tokens.push(current);
        current = '';
      }
    } else {
      current += ch;
    }
  }
  if (current) tokens.push(current);
  return tokens;
}

export function parseCommand(raw: string): ParsedCommand {
  const trimmed = raw.trim();
  if (!trimmed) return { kind: 'empty' };

  const tokens = tokenize(trimmed);
  const cmd = tokens[0].toLowerCase();

  switch (cmd) {
    case 'load': {
      if (tokens.length < 2) {
        return { kind: 'error', message: 'Usage: load <model-name>' };
      }
      return { kind: 'load', modelName: tokens[1] };
    }
    case 'run': {
      if (tokens.length < 2) {
        return { kind: 'error', message: 'Usage: run <prompt>' };
      }
      return { kind: 'run', prompt: tokens.slice(1).join(' ') };
    }
    case 'info':
      return { kind: 'info' };
    case 'contrast': {
      const val = parseFloat(tokens[1]);
      if (tokens.length < 2 || isNaN(val)) {
        return { kind: 'error', message: 'Usage: contrast <gamma> (e.g. contrast 3.5)' };
      }
      return { kind: 'contrast', gamma: val };
    }
    case 'slice': {
      if (tokens.length < 2) {
        return { kind: 'error', message: 'Usage: slice <name|depth> (auto, resid_pre, attn, delta_attn, mlp, delta_mlp, resid_post, or 0.0-1.0)' };
      }
      const arg = tokens[1].toLowerCase();
      // Check named slices first
      if (arg in SLICE_DEPTH_MAP) {
        return { kind: 'slice', depth: SLICE_DEPTH_MAP[arg] };
      }
      // Try numeric depth
      const numVal = parseFloat(arg);
      if (!isNaN(numVal)) {
        return { kind: 'slice', depth: Math.max(0, Math.min(1, numVal)) };
      }
      return { kind: 'error', message: `Unknown slice: ${tokens[1]}. Valid: ${Object.keys(SLICE_DEPTH_MAP).join(', ')}, or a number 0.0-1.0` };
    }
    case 'layout': {
      if (tokens.length < 2) {
        return { kind: 'error', message: 'Usage: layout <stack|exploded|staircase> [param] [--iso]' };
      }
      // Extract --iso flag from anywhere in tokens
      const isoFlag = tokens.some((t) => t.toLowerCase() === '--iso');
      const args = tokens.filter((t) => t.toLowerCase() !== '--iso');
      const mode = args[1]?.toLowerCase();
      if (mode !== 'stack' && mode !== 'exploded' && mode !== 'staircase') {
        return { kind: 'error', message: `Unknown layout: ${args[1]}. Valid: stack, exploded, staircase` };
      }
      const param = args.length > 2 ? parseFloat(args[2]) : undefined;
      if (param !== undefined && isNaN(param)) {
        return { kind: 'error', message: `Invalid parameter: ${args[2]}. Must be a number.` };
      }
      return { kind: 'layout', mode, param, iso: isoFlag || undefined };
    }
    case 'get': {
      if (tokens.length < 3 || tokens[1].toLowerCase() !== 'layer') {
        return { kind: 'error', message: 'Usage: get layer <N>' };
      }
      const layer = parseInt(tokens[2], 10);
      if (isNaN(layer) || layer < 0 || layer > 27) {
        return { kind: 'error', message: 'Layer must be 0–27' };
      }
      return { kind: 'get', layer };
    }
    case 'release': {
      if (tokens.length < 2) {
        return { kind: 'release', layer: null };
      }
      if (tokens[1].toLowerCase() !== 'layer' || tokens.length < 3) {
        return { kind: 'error', message: 'Usage: release [layer <N>]' };
      }
      const layer = parseInt(tokens[2], 10);
      if (isNaN(layer) || layer < 0 || layer > 27) {
        return { kind: 'error', message: 'Layer must be 0–27' };
      }
      return { kind: 'release', layer };
    }
    case 'help': {
      const topic = tokens.length > 1 ? tokens[1] : null;
      return { kind: 'help', topic };
    }
    default:
      return { kind: 'unknown', raw: trimmed };
  }
}
