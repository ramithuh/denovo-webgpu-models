import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, extname } from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url)),
  port = Number(process.env.PORT ?? 8095);
createServer(async (req, res) => {
  try {
    const pathname = new URL(req.url, "http://localhost").pathname;
    const p = resolve(
      root,
      "." +
        decodeURIComponent(pathname === "/" ? "/demo/index.html" : pathname),
    );
    if (
      !p.startsWith(root) ||
      p.includes("/.git/") ||
      p.includes("/.venv/") ||
      p.includes("/.cache/upstream/")
    )
      throw Error("Forbidden");
    const bytes = await readFile(p);
    res.writeHead(200, {
      "Content-Type":
        {
          ".js": "text/javascript",
          ".mjs": "text/javascript",
          ".html": "text/html",
          ".json": "application/json",
          ".css": "text/css",
        }[extname(p)] ?? "application/octet-stream",
    });
    res.end(bytes);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}).listen(port, "127.0.0.1", () =>
  console.log(`FrameFlow: http://127.0.0.1:${port}/`),
);
