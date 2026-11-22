import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url)),
  port = Number(process.env.PORT ?? 8096);
createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost"),
      name =
        url.pathname === "/"
          ? "/demo/index.html"
          : decodeURIComponent(url.pathname),
      p = path.resolve(root, "." + name);
    if (
      !p.startsWith(root) ||
      name.includes("/.git") ||
      !(await stat(p)).isFile()
    ) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".md": "text/plain",
      }[path.extname(p)] ?? "application/octet-stream",
    );
    createReadStream(p)
      .on("error", () => res.destroy())
      .pipe(res);
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(port, "127.0.0.1", () =>
  console.log(`Proteina demo: http://127.0.0.1:${port}/`),
);
