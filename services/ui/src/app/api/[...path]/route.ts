const API_PROXY_TARGET = (process.env.API_PROXY_TARGET || "http://api:8000").replace(/\/+$/, "");

function buildUpstreamUrl(path: string[] | undefined, requestUrl: string): string {
  const normalizedPath = Array.isArray(path) ? path.join("/") : "";
  const incoming = new URL(requestUrl);
  const suffix = incoming.search || "";
  return `${API_PROXY_TARGET}/${normalizedPath}${suffix}`;
}

function copyRequestHeaders(headers: Headers): Headers {
  const nextHeaders = new Headers(headers);
  nextHeaders.delete("host");
  nextHeaders.delete("connection");
  nextHeaders.delete("content-length");
  return nextHeaders;
}

function copyResponseHeaders(headers: Headers): Headers {
  const nextHeaders = new Headers(headers);
  nextHeaders.delete("content-encoding");
  nextHeaders.delete("transfer-encoding");
  nextHeaders.delete("connection");
  return nextHeaders;
}

async function proxy(request: Request, path: string[] | undefined): Promise<Response> {
  const upstreamUrl = buildUpstreamUrl(path, request.url);
  const method = request.method.toUpperCase();
  const init: RequestInit = {
    method,
    headers: copyRequestHeaders(request.headers),
    redirect: "manual",
  };
  if (method !== "GET" && method !== "HEAD") {
    init.body = await request.text();
  }
  const upstream = await fetch(upstreamUrl, init);
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: copyResponseHeaders(upstream.headers),
  });
}

export async function GET(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}

export async function POST(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}

export async function PUT(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}

export async function PATCH(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}

export async function DELETE(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}

export async function OPTIONS(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}

export async function HEAD(
  request: Request,
  context: { params: { path?: string[] } }
): Promise<Response> {
  return proxy(request, context.params.path);
}
