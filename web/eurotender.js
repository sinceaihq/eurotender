// EuroTender inference in plain JavaScript (browser or Node >= 18).
//
// A line-by-line port of eurotender.text.normalize and eurotender.model:
// the same regular expressions, scikit-learn's HashingVectorizer tokenisation
// (word 1-2-grams, char_wb n-grams), MurmurHash3, sublinear TF-IDF with
// per-block L2 normalisation and a linear head with temperature softmax.
// tests/test_web.py checks it against the Python implementation feature by
// feature on real notices.

import { CASEFOLD } from "./casefold-table.js";

// Python's \w, \d and \s for str patterns, spelled out (JS \w/\d/\b are ASCII).
const W = "[\\p{L}\\p{N}_]";
const D = "\\p{Nd}";
const S = "[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const B = `(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`;

const EMAIL_RE = new RegExp(
  `${B}[\\p{L}\\p{N}_.+\\-]+ ?(?:@|\\(at\\)|\\[at\\]|\\{at\\}) ?[\\p{L}\\p{N}_\\-]+(?:\\.[\\p{L}\\p{N}_\\-]+)+${B}`,
  "giu",
);
const URL_RE = new RegExp(`${B}(?:https?://|www\\.)[^${S.slice(1, -1)}<>"')]+`, "giu");
const PHONE_INTL_RE = new RegExp(
  `(?<![\\p{L}\\p{N}_+])(?:\\+|00)[1-9]${D}{0,2}(?:${S}|[./\\-])?(?:\\(0\\)(?:${S}|[./\\-])?)?${D}{1,4}` +
    `(?:(?:${S}|[./\\-])?${D}{2,4}){1,4}${B}`,
  "gu",
);
const PHONE_KEYWORD_RE = new RegExp(
  `${B}(?:tel|tél|telefon|telefono|teléfono|telefone|telephone|phone|puh|puhelin` +
    `|tlf|τηλ|тел|gsm|mobile?|mob)${B}\\.?${S}*(?:nr\\.?|no\\.?|:)?${S}*` +
    `(\\+?\\(?${D}(?:${D}|${S}|[()./\\-]){5,}${D})`,
  "giu",
);
const HONORIFIC =
  "(?:Herrn?|Frau|Mr|Mrs|Ms|Mme|Mlle|Madame|Monsieur|Mevrouw|Mevr|Dhr|Sig\\.ra|Sig" +
  "|Dott\\.ssa|Dott|Dr|Prof|Ing)\\.?";
const NAME_WORD = "[A-ZÀ-ÖØ-ÞĀ-Ž][\\p{L}\\p{N}_'\\-]+";
const NAME_RE = new RegExp(
  `${B}${HONORIFIC}(?:${S}+${HONORIFIC})*${S}+(?:[A-Z]\\.${S}*)?${NAME_WORD}(?:${S}+${NAME_WORD})?`,
  "gu",
);
const CPV_CODE_RE = new RegExp(`(?<!${D})${D}{8}(?:${S}?-${S}?${D})?(?!${D})`, "gu");
const WS_RE = new RegExp(`${S}+`, "gu");
const WS_EDGE_RE = new RegExp(`^${S}+|${S}+$`, "gu");
const SPLIT_RE = new RegExp(`${S}+`, "u");
const WORD_RE = new RegExp(`${W}{2,}`, "gu");
const ND_RE = new RegExp(D, "u");

export function pyStrip(s) {
  return s.replace(WS_EDGE_RE, "");
}

function phoneOrKeep(match) {
  let number = match.startsWith("00") ? match.slice(2) : match.replace(/^\++/u, "");
  let digits = 0;
  for (const ch of number) if (ND_RE.test(ch)) digits++;
  return digits >= 9 ? "[PHONE]" : match;
}

export function scrubPii(text) {
  if (!text) return "";
  text = text.replace(EMAIL_RE, "[EMAIL]");
  text = text.replace(URL_RE, "[URL]");
  text = text.replace(PHONE_KEYWORD_RE, (m, g1) => m.slice(0, m.length - g1.length) + "[PHONE]");
  text = text.replace(PHONE_INTL_RE, phoneOrKeep);
  text = text.replace(NAME_RE, "[NAME]");
  return text;
}

export function casefold(text) {
  let out = "";
  for (const ch of text) {
    const mapped = CASEFOLD.get(ch.codePointAt(0));
    out += mapped !== undefined ? mapped : ch.toLowerCase();
  }
  return out;
}

export function normalize(text) {
  text = (text || "").normalize("NFKC");
  text = scrubPii(text).replace(CPV_CODE_RE, " cpvcode ");
  text = casefold(text);
  return pyStrip(text.replace(WS_RE, " "));
}

export function compose(title, description = "", maxDescriptionChars = 2000) {
  title = pyStrip(title || "");
  description = Array.from(pyStrip(description || "")).slice(0, maxDescriptionChars).join("");
  return description && description !== title ? `${title}\n${description}` : title;
}

// ---------------------------------------------------------------- hashing

const encoder = new TextEncoder();

export function murmurhash3_32(bytes, seed = 0) {
  let h = seed >>> 0;
  const n = bytes.length;
  const nblocks = n >> 2;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;
  for (let i = 0; i < nblocks; i++) {
    const j = i << 2;
    let k = bytes[j] | (bytes[j + 1] << 8) | (bytes[j + 2] << 16) | (bytes[j + 3] << 24);
    k = Math.imul(k, c1);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, c2);
    h ^= k;
    h = (h << 13) | (h >>> 19);
    h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }
  let k = 0;
  const tail = nblocks << 2;
  switch (n & 3) {
    case 3:
      k ^= bytes[tail + 2] << 16;
    // falls through
    case 2:
      k ^= bytes[tail + 1] << 8;
    // falls through
    case 1:
      k ^= bytes[tail];
      k = Math.imul(k, c1);
      k = (k << 15) | (k >>> 17);
      k = Math.imul(k, c2);
      h ^= k;
  }
  h ^= n;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h | 0; // signed, like sklearn's murmurhash3_32(positive=False)
}

function bucket(feature, nFeatures) {
  const h = murmurhash3_32(encoder.encode(feature));
  if (h === -2147483648) return (2147483647 - (nFeatures - 1)) % nFeatures;
  return Math.abs(h) % nFeatures;
}

export function wordNgrams(doc, [minN, maxN]) {
  const tokens = doc.match(WORD_RE) || [];
  const out = [];
  if (minN === 1) out.push(...tokens);
  for (let n = Math.max(minN, 2); n <= Math.min(maxN, tokens.length); n++) {
    for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
  }
  return out;
}

export function charWbNgrams(doc, [minN, maxN]) {
  const out = [];
  for (const word of doc.split(SPLIT_RE)) {
    if (!word) continue;
    const w = Array.from(` ${word} `);
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      out.push(w.slice(0, n).join(""));
      while (offset + n < w.length) {
        offset++;
        out.push(w.slice(offset, offset + n).join(""));
      }
      if (offset === 0) break;
    }
  }
  return out;
}

/** Hashed raw counts of a normalised document: Map(featureIndex -> count). */
export function hashedCounts(doc, cfg) {
  const nWord = 2 ** cfg.word_bits;
  const nChar = 2 ** cfg.char_bits;
  const counts = new Map();
  for (const g of wordNgrams(doc, cfg.word_ngram_range)) {
    const i = bucket(g, nWord);
    counts.set(i, (counts.get(i) || 0) + 1);
  }
  for (const g of charWbNgrams(doc, cfg.char_ngram_range)) {
    const i = nWord + bucket(g, nChar);
    counts.set(i, (counts.get(i) || 0) + 1);
  }
  return counts;
}

// ---------------------------------------------------------------- model

function halfToFloat(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/**
 * Web model: meta (JSON) plus one binary blob laid out as described in
 * scripts/export_web.py.
 */
export class WebModel {
  constructor(meta, buffer) {
    this.meta = meta;
    this.cfg = meta.features;
    const nFeatures = 2 ** this.cfg.word_bits + 2 ** this.cfg.char_bits;
    let off = 0;
    const idf16 = new Uint16Array(buffer, off, nFeatures);
    off += nFeatures * 2;
    this.idf = new Float32Array(nFeatures);
    for (let i = 0; i < nFeatures; i++) this.idf[i] = halfToFloat(idf16[i]);
    this.heads = {};
    for (const head of meta.heads) {
      const nKept = head.n_kept;
      const nClasses = head.classes.length;
      off = head.offset;
      const ids = new Uint32Array(buffer, off, nKept);
      off += nKept * 4;
      const q = new Int8Array(buffer, off, nKept * nClasses);
      const row = new Map();
      for (let i = 0; i < nKept; i++) row.set(ids[i], i);
      this.heads[head.name] = { ...head, row, q, scale: Float32Array.from(head.scale) };
    }
  }

  static async fromUrl(base) {
    const meta = await (await fetch(`${base}/web-model.json`)).json();
    const buffer = await (await fetch(`${base}/web-model.bin`)).arrayBuffer();
    return new WebModel(meta, buffer);
  }

  features(text) {
    const counts = hashedCounts(normalize(text), this.cfg);
    const nWord = 2 ** this.cfg.word_bits;
    const entries = [];
    let wn = 0;
    let cn = 0;
    for (const [i, c] of counts) {
      const v = Math.log1p(c) * this.idf[i];
      entries.push([i, v]);
      if (i < nWord) wn += v * v;
      else cn += v * v;
    }
    const ws = wn > 0 ? Math.sqrt(1 - this.cfg.char_weight) / Math.sqrt(wn) : 0;
    const cs = cn > 0 ? Math.sqrt(this.cfg.char_weight) / Math.sqrt(cn) : 0;
    return entries.map(([i, v]) => [i, v * (i < nWord ? ws : cs)]);
  }

  predict(text, { level = "division", topK = 5 } = {}) {
    const head = this.heads[level];
    const nClasses = head.classes.length;
    const scores = Float64Array.from(head.intercept);
    for (const [i, v] of this.features(text)) {
      const r = head.row.get(i);
      if (r === undefined) continue;
      const base = r * nClasses;
      for (let c = 0; c < nClasses; c++) scores[c] += v * head.q[base + c] * head.scale[c];
    }
    let max = -Infinity;
    for (const s of scores) max = Math.max(max, s / head.temperature);
    const exps = Array.from(scores, (s) => Math.exp(s / head.temperature - max));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps
      .map((e, c) => ({ code: head.classes[c], score: e / total }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}
