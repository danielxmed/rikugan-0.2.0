import type { WsIncoming, WsOutgoing } from '../types/messages';

type MessageHandler = (msg: WsIncoming) => void;
type BinaryHandler = (data: ArrayBuffer) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private handlers = new Set<MessageHandler>();
  private binaryHandlers = new Set<BinaryHandler>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private url: string;

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      console.log('[ws] connected');
    };

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        for (const handler of this.binaryHandlers) {
          handler(event.data);
        }
        return;
      }

      try {
        const msg: WsIncoming = JSON.parse(event.data);
        for (const handler of this.handlers) {
          handler(msg);
        }
      } catch (e) {
        console.error('[ws] failed to parse message', e);
      }
    };

    this.ws.onclose = () => {
      console.log('[ws] disconnected, reconnecting in 2s...');
      this.scheduleReconnect();
    };

    this.ws.onerror = (e) => {
      console.error('[ws] error', e);
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  send(msg: WsOutgoing): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[ws] not connected, dropping message');
      return;
    }
    this.ws.send(JSON.stringify(msg));
  }

  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  onBinary(handler: BinaryHandler): () => void {
    this.binaryHandlers.add(handler);
    return () => {
      this.binaryHandlers.delete(handler);
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}

const wsUrl =
  location.protocol === 'https:'
    ? `wss://${location.host}/ws`
    : `ws://${location.host}/ws`;

export const wsService = new WebSocketService(wsUrl);
