import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url)),
  port = Number(process.env.PORT ?? 8092);
createServer((req, res) => {
  const u = new URL(req.url, "http://localhost"),
    p = path.resolve(
      root,
      "." +
        decodeURIComponent(
          u.pathname === "/" ? "/demo/index.html" : u.pathname,
        ),
    );
  if (!p.startsWith(root) || u.pathname.includes("/.")) {
    res.writeHead(403).end();
    return;
  }
  res.setHeader(
    "Content-Type",
    {
      ".js": "text/javascript",
      ".mjs": "text/javascript",
      ".json": "application/json",
      ".html": "text/html",
    }[path.extname(p)] ?? "application/octet-stream",
  );
  createReadStream(p)
    .on("error", () => res.writeHead(404).end())
    .pipe(res);
}).listen(port, "127.0.0.1", () =>
  console.log(`FoldFlow 2: http://localhost:${port}`),
);
