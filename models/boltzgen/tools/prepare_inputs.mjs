import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createFeatures } from "../shared/input.js";
import { randomInputs } from "../shared/random.js";
import { cases, recycledMotif } from "../test/cases.js";
await mkdir(new URL("../weights/dynamic/", import.meta.url), {
  recursive: true,
});
for (const c of [...cases, recycledMotif]) {
  const opts = { ...c };
  if (c.targetFile)
    opts.targetPDB = await readFile(
      new URL("../" + c.targetFile, import.meta.url),
      "utf8",
    );
  const f = createFeatures(opts),
    rng = randomInputs(f.get("f:atom_pad_mask").shape[1], c.numSteps, c.seed);
  const data = Object.fromEntries(
    [...f, ...rng].map(([k, t]) => [
      k,
      { shape: t.shape, data: Array.from(t.data) },
    ]),
  );
  await writeFile(
    new URL("../weights/dynamic/" + c.name + ".json", import.meta.url),
    JSON.stringify({ options: c, features: data }),
  );
}
