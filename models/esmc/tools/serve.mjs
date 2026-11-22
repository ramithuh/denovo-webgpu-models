/** Static localhost demo server, always an OS-selected ephemeral port. */
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://localhost"),
    pathname =
      url.pathname === "/"
        ? "/demo/index.html"
        : decodeURIComponent(url.pathname),
    p = path.resolve(root, "." + pathname);
  if (!p.startsWith(root) || pathname.includes("/.")) {
    res.writeHead(404);
    res.end();
    return;
  }
  res.setHeader(
    "Content-Type",
    {
      ".js": "text/javascript",
      ".html": "text/html",
      ".json": "application/json",
      ".md": "text/plain",
    }[path.extname(p)] ?? "application/octet-stream",
  );
  const stream = createReadStream(p);
  stream.on("error", () => {
    res.writeHead(404);
    res.end();
  });
  stream.pipe(res);
});
server.listen(0, "127.0.0.1", () =>
  console.log(`ESMC demo: http://127.0.0.1:${server.address().port}/`),
);
