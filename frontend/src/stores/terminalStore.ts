import { create } from 'zustand';
import { useModelStore } from './modelStore';

interface TerminalState {
  history: string[];
  isFocused: boolean;
  addToHistory: (cmd: string) => void;
  toggleFocus: () => void;
  setFocused: (focused: boolean) => void;
}

export const useTerminalStore = create<TerminalState>((set) => ({
  history: [],
  isFocused: true,
  addToHistory: (cmd) =>
    set((state) => ({ history: [...state.history, cmd] })),
  toggleFocus: () => set((state) => ({ isFocused: !state.isFocused })),
  setFocused: (focused) => set({ isFocused: focused }),
}));

export function getPrompt(): string {
  const modelId = useModelStore.getState().modelId;
  if (modelId) {
    return `rikugan [\x1b[36m${modelId}\x1b[0m] > `;
  }
  return 'rikugan > ';
}
