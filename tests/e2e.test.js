// VeriGraph end-to-end test suite (Node, no framework).
// Run: node tests/e2e.test.js [https://your-worker.workers.dev]
// Mirrors ReproGraph's "20 passing tests" claim with a single self-test +
// a live pipeline run + structured assertions.

const BASE = process.argv[2] || "https://verigraph.arechampionw.workers.dev";
let passed = 0, failed = 0;
const results = [];

async function check(name, cond) {
  if (cond) { passed++; results.push(`  PASS  ${name}`); }
  else { failed++; results.push(`  FAIL  ${name}`); }
}

async function main() {
  console.log(`VeriGraph e2e tests -> ${BASE}\n`);

  // 1. Health + model chains
  const h = await (await fetch(`${BASE}/api/health`)).json();
  await check("health endpoint ok", h.ok === true);
  await check("vision model chain has >=2 models", h.chains?.vision?.length >= 2);
  await check("text model chain has >=3 models", h.chains?.text?.length >= 3);

  // 2. Self-test endpoint (8 built-in assertions)
  const t = await (await fetch(`${BASE}/api/test`)).json();
  await check("self-test returns ok", t.ok === true);
  await check("self-test passes all built-in checks", t.passed === t.total);
  await check("self-test score >= 70", t.score?.total >= 70);
  await check("self-test cost < $0.05", t.cost_usd < 0.05);
  for (const c of t.checks || []) await check(`self-test: ${c.name}`, c.pass);

  // 3. Live pipeline run with a known figure + citations
  const fs = require("fs");
  const path = require("path");
  const figPath = path.join(__dirname, "..", "public", "demo-figure.png");
  const b64 = fs.existsSync(figPath) ? fs.readFileSync(figPath).toString("base64") : "";
  const body = {
    paper_title: "Test paper",
    paper_text:
      "Abstract. We benchmark vision-language models on figure parsing. Kimi K2.7 reaches 91.2% accuracy at 340 ms; GLM 5.2 reaches 88.7% at 210 ms. Vaswani et al. (2017) introduced self-attention.",
    figures: [{ name: "table1.png", image_base64: b64, mime_type: "image/png" }],
    citations: [
      { id: "C1", raw_text: "Vaswani, A. et al. Attention is all you need. NeurIPS 2017.", claim: "Self-attention enables parallelizable sequence modeling." },
      { id: "C2", raw_text: "Vortex, P. et al. Hyperdimensional diffusion transformers for figure parsing. Nature Machine Intelligence 9 (2025).", claim: "Diffusion parsers beat autoregressive ones." },
    ],
  };
  const r = await (await fetch(`${BASE}/api/analyze`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  })).json();
  await check("analyze returns extracted_data", Array.isArray(r.extracted_data) && r.extracted_data.length >= 1);
  await check("vision extracted rows from the table", r.extracted_data?.[0]?.rows?.length >= 1);
  await check("real citation C1 verified", r.citation_verdicts?.some(v => v.citation_id === "C1" && v.status === "verified"));
  await check("fabricated citation C2 flagged", r.citation_verdicts?.some(v => v.citation_id === "C2" && v.status !== "verified"));
  await check("hypotheses generated", r.hypotheses?.length >= 1);
  await check("at least one SUPPORTED hypothesis", r.hypotheses?.some(h => h.classification === "SUPPORTED"));
  await check("hypotheses cite evidence pointers", r.hypotheses?.some(h => h.evidence_pointers?.length > 0));
  await check("handoff graph present", Array.isArray(r.handoffs) && r.handoffs.length >= 2);
  await check("trace has >=3 distinct agents", new Set(r.trace?.map(t => t.agent)).size >= 3);
  await check("cost table itemized per agent", r.cost_table?.length >= 3);
  await check("total cost < $0.05", r.total_cost_usd < 0.05);
  await check("composite score computed (0-100)", typeof r.score?.total === "number" && r.score.total >= 0 && r.score.total <= 100);
  await check("research_hours_saved estimated", typeof r.research_hours_saved === "number");
  await check("markdown report generated", typeof r.report_markdown === "string" && r.report_markdown.includes("VeriGraph report"));

  // 4. Link-fetch endpoint
  const fr = await (await fetch(`${BASE}/api/fetch?url=${encodeURIComponent("https://example.com")}`)).json();
  await check("link fetch returns content", fr.kind === "html" || fr.kind === "pdf" || fr.kind === "image" || (fr.text && fr.text.length > 0));

  console.log(results.join("\n"));
  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
}
main().catch(e => { console.error(e); process.exit(1); });
