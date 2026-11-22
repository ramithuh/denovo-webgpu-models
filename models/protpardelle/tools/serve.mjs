import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, extname } from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url));
const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(
        new URL(req.url, "http://localhost").pathname,
      ),
      file = resolve(root, "." + path);
    if (
      !file.startsWith(root) ||
      (!/^\/(demo|shared|cpu|webgpu|weights)\//.test(path) &&
        path !== "/README_WEBGPU.md")
    )
      throw new Error("Not found");
    const bytes = await readFile(
      file.endsWith("/") ? file + "index.html" : file,
    );
    res.writeHead(200, {
      "Content-Type":
        {
          ".html": "text/html",
          ".js": "text/javascript",
          ".json": "application/json",
          ".md": "text/plain",
        }[extname(file)] ?? "application/octet-stream",
    });
    res.end(bytes);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});
server.listen(0, "127.0.0.1", () =>
  console.log(
    `Local: http://127.0.0.1:${server.address().port}/demo/index.html`,
  ),
);
for (const signal of ["SIGINT", "SIGTERM"])
  process.on(signal, () => server.close());
