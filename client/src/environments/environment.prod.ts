export const environment = {
  production: true,
  // GitHub Pages is HTTPS-only, so apiBase must be https -- the WebSocket URL
  // is derived from it in http-inference-api.ts, which gives wss for free.
  useMock: false,
  apiBase: 'https://mdebug-server-vutzb66e2a-uc.a.run.app',
};
