import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(
        new URL(req.url, "http://localhost").pathname,
      ),
      file = path.resolve(root, "." + pathname);
    if (!file.startsWith(root) || pathname.includes("/.")) {
      res.writeHead(403);
      res.end();
      return;
    }
    const p = (await stat(file)).isDirectory()
      ? path.join(file, "index.html")
      : file;
    res.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".css": "text/css",
      }[path.extname(p)] ?? "application/octet-stream",
    );
    createReadStream(p)
      .on("error", () => res.destroy())
      .pipe(res);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});
server.listen(Number(process.env.PORT ?? 8080), "127.0.0.1", () =>
  console.log(`Demo: http://127.0.0.1:${server.address().port}/demo/`),
);
