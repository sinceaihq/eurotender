// Reads JSON lines {"text": ...} on stdin, writes {"norm": ..., "counts": [[i, c], ...]}.
// Used by tests/test_web.py to compare with the Python featurizer.
import { createInterface } from "node:readline";
import { normalize, hashedCounts } from "./eurotender.js";

const cfg = JSON.parse(process.argv[2]);
const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line) continue;
  const { text } = JSON.parse(line);
  const norm = normalize(text);
  const counts = [...hashedCounts(norm, cfg)].sort((a, b) => a[0] - b[0]);
  process.stdout.write(JSON.stringify({ norm, counts }) + "\n");
}
