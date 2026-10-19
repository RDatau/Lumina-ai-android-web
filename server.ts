import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;
  const isProduction = process.env.NODE_ENV === 'production';

  app.use(cors());

  // Mount raw body parser khusus /api/hf-proxy SEBELUM parser umum json/urlencoded
  // agar form-data multipart (file upload Gradio) dan binary stream diteruskan secara utuh
  app.use('/api/hf-proxy', express.raw({ type: '*/*', limit: '100mb' }));

  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ limit: '50mb', extended: true }));

  // ==========================================
  // HF Proxy: Bypass CORS untuk Hugging Face Space & ZeroGPU
  // ==========================================
  app.all('/api/hf-proxy', async (req, res) => {
    const rawTargetUrl = (req.query.url as string) || (req.headers['x-target-url'] as string);
    if (!rawTargetUrl) {
      return res.status(400).json({ error: 'Missing target URL in query or x-target-url header' });
    }

    try {
      let targetUrl = rawTargetUrl.trim();
      try {
        new URL(targetUrl);
      } catch {
        targetUrl = decodeURIComponent(rawTargetUrl).trim();
      }

      // Validasi protokol dan hostname
      let parsedUrl: URL;
      try {
        parsedUrl = new URL(targetUrl);
      } catch (urlErr) {
        return res.status(400).json({ error: `URL target tidak valid: "${targetUrl}"` });
      }

      if (!parsedUrl.protocol.startsWith('http') || !parsedUrl.hostname || parsedUrl.hostname === 'https' || parsedUrl.hostname === 'http' || !parsedUrl.hostname.includes('.')) {
        return res.status(400).json({ error: `Hostname URL target tidak valid: "${parsedUrl.hostname}". Pastikan URL lengkap.` });
      }

      const forwardHeaders: Record<string, string> = {};

      if (req.headers['authorization']) {
        forwardHeaders['authorization'] = req.headers['authorization'] as string;
      }
      if (req.headers['content-type']) {
        forwardHeaders['content-type'] = req.headers['content-type'] as string;
      }
      if (req.headers['accept']) {
        forwardHeaders['accept'] = req.headers['accept'] as string;
      }

      const fetchOptions: any = {
        method: req.method,
        headers: forwardHeaders,
        redirect: 'follow',
      };

      if (req.method !== 'GET' && req.method !== 'HEAD') {
        fetchOptions.duplex = 'half';
        if (Buffer.isBuffer(req.body) && req.body.length > 0) {
          fetchOptions.body = req.body;
          forwardHeaders['content-length'] = String(req.body.length);
        } else if (req.body && typeof req.body === 'object' && Object.keys(req.body).length > 0) {
          fetchOptions.body = JSON.stringify(req.body);
          if (!forwardHeaders['content-type']) {
            forwardHeaders['content-type'] = 'application/json';
          }
          forwardHeaders['content-length'] = String(Buffer.byteLength(fetchOptions.body));
        } else if (typeof req.body === 'string' && req.body.length > 0) {
          fetchOptions.body = req.body;
          forwardHeaders['content-length'] = String(Buffer.byteLength(req.body));
        }
      }

      const upstreamResp = await fetch(targetUrl, fetchOptions);

      // Forward status
      res.status(upstreamResp.status);

      // Forward response headers
      const contentType = upstreamResp.headers.get('content-type');
      if (contentType) {
        res.setHeader('content-type', contentType);
      }

      // Handle Server-Sent Events (SSE) streaming (/call/.../event_id)
      if (contentType && contentType.includes('text/event-stream')) {
        res.setHeader('cache-control', 'no-cache');
        res.setHeader('connection', 'keep-alive');
        res.setHeader('content-type', 'text/event-stream; charset=utf-8');
        res.flushHeaders?.();

        if (upstreamResp.body) {
          const reader = upstreamResp.body.getReader();
          req.on('close', () => {
            reader.cancel().catch(() => {});
          });
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              res.write(value);
            }
          } catch (streamErr) {
            console.warn('[HF-Proxy] Stream interrupted:', streamErr);
          } finally {
            res.end();
          }
          return;
        }
      }

      // Send standard response (JSON, Text, or Binary/Image)
      const arrayBuffer = await upstreamResp.arrayBuffer();
      res.send(Buffer.from(arrayBuffer));

    } catch (err: any) {
      console.error('[HF-Proxy Error]', err);
      res.status(502).json({
        error: 'Proxy to Hugging Face Space failed',
        message: err?.message || String(err)
      });
    }
  });

  // ==========================================
  // Download Proxy: Force Content-Disposition for Android WebView
  // ==========================================
  // ==========================================
  // Download Proxy: Force Content-Disposition for Android WebView
  // ==========================================
  // ==========================================
  // Download Proxy: Force Content-Disposition for Android WebView
  // ==========================================
  // ==========================================
  // Download Proxy: Force Content-Disposition for Android WebView
  // ==========================================
  app.post('/api/download-proxy', express.json({ limit: '50mb' }), (req, res) => {
    const { data, fileName, mimeType } = req.body;
    if (!data || !fileName) {
      console.error('Missing data or fileName in POST body', { bodyKeys: Object.keys(req.body) });
      return res.status(400).send('Missing data or fileName');
    }
    
    try {
      // Remove data URL prefix if present
      const base64Data = typeof data === 'string' ? data.replace(/^data:.*,/, '') : data;
      const buffer = Buffer.from(base64Data, 'base64');
      
      res.setHeader('Content-Type', (mimeType as string) || 'application/octet-stream');
      res.setHeader('Content-Disposition', `attachment; filename="${fileName.replace(/"/g, '\\"')}"`);
      res.send(buffer);
    } catch (e) {
      console.error('Proxy error:', e);
      res.status(500).send('Internal Server Error');
    }
  });

  // ==========================================
  // Vite Middlewares (Dev) or Static dist (Prod)
  // ==========================================
  if (!isProduction) {
    const vite = await createViteServer({
      server: { 
        middlewareMode: true,
        host: '0.0.0.0',
        port: PORT
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Lumina AI Server listening on port ${PORT} (0.0.0.0:${PORT})`);
  });
}

startServer().catch(err => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
