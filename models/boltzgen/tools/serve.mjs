import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { resolve, extname } from "node:path";
const root = resolve(import.meta.dirname, ".."),
  port = Number(process.argv[2] ?? 8080);
createServer(async (req, res) => {
  try {
    let file = resolve(
      root,
      "." + decodeURIComponent(new URL(req.url, "http://localhost").pathname),
    );
    if (!file.startsWith(root + "/")) throw Error();
    if ((await stat(file)).isDirectory()) file += "/index.html";
    if (file.includes("/.cache/") || file.includes("/.venv/")) throw Error();
    res.setHeader(
      "Content-Type",
      {
        ".html": "text/html",
        ".js": "text/javascript",
        ".json": "application/json",
        ".md": "text/plain",
      }[extname(file)] ?? "application/octet-stream",
    );
    createReadStream(file).pipe(res);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
}).listen(port, "127.0.0.1", () =>
  console.log(`http://127.0.0.1:${port}/demo/`),
);
