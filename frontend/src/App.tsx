import { useEffect } from 'react';
import './App.css';
import Viewport from './components/viewport/Viewport';
import TerminalPanel from './components/terminal/Terminal';
import { wsService } from './services/websocket';
import { useTerminalStore } from './stores/terminalStore';

function App() {
  useEffect(() => {
    wsService.connect();
    return () => wsService.disconnect();
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '`' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        useTerminalStore.getState().toggleFocus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <div className="app-layout">
      <div className="viewport-panel">
        <Viewport />
      </div>
      <div className="terminal-panel">
        <TerminalPanel />
      </div>
    </div>
  );
}

export default App;
