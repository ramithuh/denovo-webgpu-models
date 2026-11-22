import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url));
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost"),
      name = decodeURIComponent(url.pathname),
      file = path.resolve(
        root,
        "." + (name.endsWith("/") ? name + "index.html" : name),
      );
    if (
      !file.startsWith(root) ||
      name.includes("/.git") ||
      name.includes("/.venv")
    )
      throw new Error("Invalid path");
    if (!(await stat(file)).isFile()) throw new Error("Not a file");
    res.setHeader(
      "Content-Type",
      {
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".html": "text/html",
        ".json": "application/json",
        ".css": "text/css",
      }[path.extname(file)] ?? "application/octet-stream",
    );
    createReadStream(file)
      .on("error", () => res.destroy())
      .pipe(res);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});
server.listen(Number(process.env.PORT ?? 8080), "127.0.0.1", () =>
  console.log(
    `SiD-Protein demo: http://127.0.0.1:${server.address().port}/demo/`,
  ),
);
