export const config = {
  runtime: 'edge',
};

export default async function handler(req: Request) {
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': '*',
      },
    });
  }

  const reqUrl = new URL(req.url);
  const rawTargetUrl = reqUrl.searchParams.get('url') || req.headers.get('x-target-url');

  if (!rawTargetUrl) {
    return new Response(JSON.stringify({ error: 'Missing target URL in query (?url=...) or x-target-url header' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }

  let targetUrl = rawTargetUrl.trim();
  try {
    new URL(targetUrl);
  } catch {
    targetUrl = decodeURIComponent(rawTargetUrl).trim();
  }

  let parsedUrl: URL;
  try {
    parsedUrl = new URL(targetUrl);
  } catch {
    return new Response(JSON.stringify({ error: `URL target tidak valid: "${targetUrl}"` }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }

  if (!parsedUrl.protocol.startsWith('http') || !parsedUrl.hostname || !parsedUrl.hostname.includes('.')) {
    return new Response(JSON.stringify({ error: `Hostname target tidak valid: "${parsedUrl.hostname}"` }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }

  const forwardHeaders = new Headers();
  const auth = req.headers.get('authorization');
  if (auth) forwardHeaders.set('authorization', auth);

  const contentType = req.headers.get('content-type');
  if (contentType) forwardHeaders.set('content-type', contentType);

  const accept = req.headers.get('accept');
  if (accept) forwardHeaders.set('accept', accept);

  const fetchOptions: any = {
    method: req.method,
    headers: forwardHeaders,
    redirect: 'follow',
  };

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    fetchOptions.body = req.body;
    fetchOptions.duplex = 'half';
  }

  try {
    const upstreamResp = await fetch(targetUrl, fetchOptions);

    const responseHeaders = new Headers();
    responseHeaders.set('Access-Control-Allow-Origin', '*');

    const upstreamContentType = upstreamResp.headers.get('content-type');
    if (upstreamContentType) {
      responseHeaders.set('content-type', upstreamContentType);
    }

    if (upstreamContentType && upstreamContentType.includes('text/event-stream')) {
      responseHeaders.set('cache-control', 'no-cache');
      responseHeaders.set('connection', 'keep-alive');
    }

    return new Response(upstreamResp.body, {
      status: upstreamResp.status,
      headers: responseHeaders,
    });
  } catch (err: any) {
    return new Response(JSON.stringify({ error: 'Proxy to Hugging Face Space failed', message: err?.message || String(err) }), {
      status: 502,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }
}
