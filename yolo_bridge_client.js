export function createYoloHttpBridge(options = {}) {
  const endpoint = options.endpoint || window.localStorage.getItem('mole-arena-yolo-endpoint') || 'http://127.0.0.1:8765/api/detections';
  const pollMs = options.pollMs || 250;
  let timer = null;

  async function poll() {
    try {
      const response = await fetch(endpoint, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      window.postMessage(
        {
          type: 'YOLO_BRIDGE_STATUS',
          state: payload.state || 'external-ready',
          protocol: payload.protocol || 'http-poll',
        },
        '*'
      );
      window.postMessage(
        {
          type: 'YOLO_DETECTIONS',
          detections: Array.isArray(payload.detections) ? payload.detections : [],
        },
        '*'
      );
    } catch (error) {
      window.postMessage(
        {
          type: 'YOLO_BRIDGE_STATUS',
          state: 'bridge-offline',
          protocol: 'http-poll',
        },
        '*'
      );
    }
  }

  return {
    start() {
      if (timer) {
        return;
      }
      poll();
      timer = window.setInterval(poll, pollMs);
    },
    stop() {
      if (!timer) {
        return;
      }
      window.clearInterval(timer);
      timer = null;
    },
  };
}
