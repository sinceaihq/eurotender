// Reads JSON lines {"text": ...} on stdin and prints the web model's top-5
// division codes and scores. Used by scripts/evaluate_web.py.
import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { WebModel } from "./eurotender.js";

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(`${dir}/web-model.json`, "utf8"));
const buf = readFileSync(`${dir}/web-model.bin`);
const model = new WebModel(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
const t0 = performance.now();
let n = 0;
for await (const line of rl) {
  if (!line) continue;
  const preds = model.predict(JSON.parse(line).text, { level: "division", topK: 5 });
  process.stdout.write(JSON.stringify(preds.map((p) => [p.code, p.score])) + "\n");
  n++;
}
process.stderr.write(JSON.stringify({ n, ms_per_text: (performance.now() - t0) / Math.max(n, 1) }) + "\n");
