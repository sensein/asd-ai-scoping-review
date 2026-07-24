#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (!key.startsWith("--")) continue;
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key.slice(2)] = true;
    } else {
      args[key.slice(2)] = next;
      i += 1;
    }
  }
  return args;
}

function safeTextName(recordId) {
  return `${String(recordId || "record").replace(/[^A-Za-z0-9_.-]+/g, "_")}.txt`;
}

async function extractPdfText(pdfPath, maxPages) {
  const data = new Uint8Array(await fs.readFile(pdfPath));
  const loadingTask = pdfjsLib.getDocument({
    data,
    useWorkerFetch: false,
    isEvalSupported: false,
    disableFontFace: true,
  });
  const pdf = await loadingTask.promise;
  const pageLimit = maxPages > 0 ? Math.min(pdf.numPages, maxPages) : pdf.numPages;
  const pages = [];
  for (let pageNo = 1; pageNo <= pageLimit; pageNo += 1) {
    const page = await pdf.getPage(pageNo);
    const content = await page.getTextContent();
    pages.push(content.items.map((item) => item.str || "").join(" "));
  }
  return {
    pageCount: pdf.numPages,
    extractedPages: pageLimit,
    text: pages.join("\n\n"),
  };
}

const args = parseArgs(process.argv);
const inputJson = args["input-json"];
const outputJsonl = args["output-jsonl"];
const textDir = args["text-dir"];
const maxPages = Number.parseInt(args["max-pages"] || "0", 10);

if (!inputJson || !outputJsonl || !textDir) {
  console.error("Usage: node 09_extract_pdf_texts_pdfjs.mjs --input-json records.json --output-jsonl extracted.jsonl --text-dir extracted_texts [--max-pages 0]");
  process.exit(2);
}

await fs.mkdir(textDir, { recursive: true });
const records = JSON.parse(await fs.readFile(inputJson, "utf8"));
const lines = [];

for (let index = 0; index < records.length; index += 1) {
  const record = records[index];
  const recordId = record.record_id || `ROW${String(index + 1).padStart(4, "0")}`;
  const pdfPath = record.resolved_pdf_path;
  const result = {
    record_id: recordId,
    title: record.title || "",
    pdf_path: pdfPath || "",
    extraction_status: "not_run",
    extraction_error: "",
    page_count: "",
    extracted_pages: "",
    text_char_count: 0,
    text_path: "",
  };

  try {
    if (!pdfPath) {
      throw new Error("No resolved PDF path");
    }
    const extracted = await extractPdfText(pdfPath, maxPages);
    const textPath = path.join(textDir, safeTextName(recordId));
    await fs.writeFile(textPath, extracted.text, "utf8");
    result.extraction_status = extracted.text.trim().length ? "extracted" : "empty_text";
    result.page_count = extracted.pageCount;
    result.extracted_pages = extracted.extractedPages;
    result.text_char_count = extracted.text.length;
    result.text_path = textPath;
  } catch (error) {
    result.extraction_status = "failed";
    result.extraction_error = error?.message || String(error);
  }

  lines.push(JSON.stringify(result));
  if ((index + 1) % 10 === 0 || index + 1 === records.length) {
    console.log(`Extracted ${index + 1}/${records.length}`);
  }
}

await fs.writeFile(outputJsonl, `${lines.join("\n")}\n`, "utf8");
