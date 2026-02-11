import { useEffect, useRef } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { parseCommand } from './commandParser';
import { dispatchCommand } from './commandDispatcher';
import { getPrompt } from '../../stores/terminalStore';
import { useTerminalStore } from '../../stores/terminalStore';
import { wsService } from '../../services/websocket';
import { useActivationStore } from '../../stores/activationStore';
import { useViewportStore } from '../../stores/viewportStore';
import type { WsIncoming } from '../../types/messages';

export function useTerminal(containerRef: React.RefObject<HTMLDivElement | null>) {
  const termRef = useRef<Terminal | null>(null);
  const inputBuffer = useRef('');
  const historyIndex = useRef(-1);
  const busyRef = useRef(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const term = new Terminal({
      theme: {
        background: '#0a0a0f',
        foreground: '#c0c0c0',
        cursor: '#c0c0c0',
        selectionBackground: '#264f78',
      },
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
      fontSize: 13,
      cursorBlink: true,
      convertEol: true,
      allowProposedApi: true,
    });

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.open(container);
    fitAddon.fit();
    termRef.current = term;

    const writeLine = (text: string) => {
      term.writeln(text);
    };

    const writePrompt = () => {
      term.write(getPrompt());
    };

    // WS JSON message handler
    const unsubscribeJson = wsService.onMessage((msg: WsIncoming) => {
      if (msg.type === 'activation.frame') {
        const { block_heat, num_layers, prompt } = msg.payload;
        useActivationStore.getState().setFrame(block_heat, num_layers, prompt);
      } else if (msg.type === 'activation.slices') {
        const { seq_len, d_model } = msg.payload;
        useActivationStore.getState().setSliceMeta({ seq_len, d_model });
      } else if (msg.type === 'activation.projections') {
        useActivationStore.getState().setProjMeta(msg.payload);
      } else if (msg.type === 'inference.result') {
        term.writeln(`\x1b[32m${msg.payload.text}\x1b[0m`);
        writePrompt();
      } else if (msg.type === 'error') {
        term.writeln(`\x1b[31mWS Error: ${msg.payload.message}\x1b[0m`);
        writePrompt();
      }
    });

    // WS binary handler — tagged binary frames
    const unsubscribeBinary = wsService.onBinary((data: ArrayBuffer) => {
      if (data.byteLength < 4) return;
      const tag = new Uint32Array(data, 0, 1)[0];
      const payload = data.slice(4);

      switch (tag) {
        case 0x01: {
          // Slice data
          useActivationStore.getState().setSliceData(new Float32Array(payload));
          break;
        }
        case 0x02: {
          // Projection data: token_proj + dim_proj concatenated
          const meta = useActivationStore.getState().projMeta;
          if (meta) {
            const tokenProjFloats = meta.token_proj_size / 4;
            useActivationStore.getState().setTokenProj(
              new Float32Array(payload, 0, tokenProjFloats),
            );
            useActivationStore.getState().setDimProj(
              new Float32Array(payload, meta.token_proj_size),
            );
          }
          break;
        }
      }
    });

    // Subscribe to boundary layer changes
    let prevBoundaryLayer: number | null = null;
    const unsubscribeBoundary = useViewportStore.subscribe(
      (state) => {
        if (state.boundaryLayer !== null && state.boundaryLayer !== prevBoundaryLayer) {
          prevBoundaryLayer = state.boundaryLayer;
          term.writeln(
            `\x1b[33mlayer ${state.boundaryLayer} \u2014 meso view available in Phase 2\x1b[0m`
          );
        } else if (state.boundaryLayer === null) {
          prevBoundaryLayer = null;
        }
      },
    );

    // Welcome
    term.writeln('\x1b[1mRikugan v0.2.0\x1b[0m \u2014 Mechanistic Interpretability Visualizer');
    term.writeln('Type \x1b[1mhelp\x1b[0m for commands.\r\n');
    writePrompt();

    // Key handler
    term.onKey(({ key, domEvent }) => {
      if (busyRef.current && domEvent.key !== 'c') return;

      const code = domEvent.keyCode;
      const history = useTerminalStore.getState().history;

      if (domEvent.key === 'Enter') {
        term.write('\r\n');
        const raw = inputBuffer.current;
        inputBuffer.current = '';
        historyIndex.current = -1;

        if (raw.trim()) {
          useTerminalStore.getState().addToHistory(raw);
        }

        const cmd = parseCommand(raw);
        if (cmd.kind === 'empty') {
          writePrompt();
          return;
        }

        // For 'run', prompt comes back via WS handler
        if (cmd.kind === 'run') {
          busyRef.current = true;
          dispatchCommand(cmd, writeLine).finally(() => {
            busyRef.current = false;
          });
          return;
        }

        busyRef.current = true;
        dispatchCommand(cmd, writeLine).finally(() => {
          busyRef.current = false;
          writePrompt();
        });
      } else if (domEvent.key === 'Backspace') {
        if (inputBuffer.current.length > 0) {
          inputBuffer.current = inputBuffer.current.slice(0, -1);
          term.write('\b \b');
        }
      } else if (code === 38) {
        // Up arrow
        if (history.length === 0) return;
        if (historyIndex.current === -1) {
          historyIndex.current = history.length - 1;
        } else if (historyIndex.current > 0) {
          historyIndex.current--;
        }
        clearInput(term);
        inputBuffer.current = history[historyIndex.current];
        term.write(inputBuffer.current);
      } else if (code === 40) {
        // Down arrow
        if (historyIndex.current === -1) return;
        if (historyIndex.current < history.length - 1) {
          historyIndex.current++;
          clearInput(term);
          inputBuffer.current = history[historyIndex.current];
          term.write(inputBuffer.current);
        } else {
          historyIndex.current = -1;
          clearInput(term);
          inputBuffer.current = '';
        }
      } else if (key.length === 1 && !domEvent.ctrlKey && !domEvent.altKey && !domEvent.metaKey) {
        inputBuffer.current += key;
        term.write(key);
      }
    });

    // Resize observer
    const ro = new ResizeObserver(() => {
      fitAddon.fit();
    });
    ro.observe(container);

    return () => {
      unsubscribeJson();
      unsubscribeBinary();
      unsubscribeBoundary();
      ro.disconnect();
      term.dispose();
      termRef.current = null;
    };
  }, [containerRef]);

  return termRef;
}

function clearInput(term: Terminal) {
  term.write('\x1b[2K\r');
  term.write(getPrompt());
}
