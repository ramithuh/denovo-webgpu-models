import { parsePDB, writePDB, parseMMCIF, writeMMCIF } from "../shared/structure.js";

self.onmessage = async ({ data: { input, format } }) => {
  try {
    const text = await input.text();
    const rows = /\.(cif|mmcif)$/i.test(input.name) ? parseMMCIF(text) : parsePDB(text);
    const content = format === "pdb" ? writePDB(rows) : writeMMCIF(rows);
    postMessage({ count: rows.length, blob: new Blob([content], { type: "text/plain" }) });
  } catch (error) {
    postMessage({ error: error.message });
  }
};
