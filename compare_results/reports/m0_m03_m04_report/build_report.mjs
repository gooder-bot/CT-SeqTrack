import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const pluginRoot = process.env.DATA_ANALYTICS_PLUGIN_ROOT || join(
  homedir(),
  ".codex",
  "plugins",
  "cache",
  "openai-curated-remote",
  "data-analytics",
  "0.2.8-13ceeea1f599",
);
const reportScripts = join(pluginRoot, "skills", "build-report", "scripts");
const { buildPortableArtifact } = await import(pathToFileURL(join(reportScripts, "build_portable_artifact.mjs")).href);
const { extractPortableChartSvgs } = await import(pathToFileURL(join(reportScripts, "extract_portable_chart_svgs.mjs")).href);
const { verifyPortableArtifact } = await import(pathToFileURL(join(reportScripts, "verify_portable_artifact.mjs")).href);

const reportDirectory = dirname(fileURLToPath(import.meta.url));
const artifactPath = join(reportDirectory, "artifact.json");
const reportPath = join(reportDirectory, "report.html");
const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));

let html = buildPortableArtifact(artifact);
writeFileSync(reportPath, html, "utf8");

const staticCharts = await extractPortableChartSvgs({
  actionTimeoutMs: 5000,
  htmlPath: reportPath,
  readyTimeoutMs: 15000,
});
html = buildPortableArtifact(artifact, { staticCharts });

// The packaged reader top bar uses 100vw. On Chromium, a vertical scrollbar
// reduces document.clientWidth and makes that bar overflow by half the scrollbar
// width on each side. This style changes only the portable shell width; the
// canonical artifact payload, chart data, narrative, and source metadata remain
// byte-for-byte those produced by buildPortableArtifact.
const portableLayoutCompatibility = [
  '<style data-ct-seqtrack-portable-layout-compatibility="true">',
  '.analytics-top-bar{width:100%!important;max-width:100%!important;',
  'margin-left:0!important;margin-right:0!important;left:0!important;',
  'right:auto!important;transform:none!important;box-sizing:border-box!important}',
  '.analytics-top-bar-title{min-width:0!important;overflow:hidden!important;text-overflow:ellipsis!important}',
  '</style>',
].join('');
html = html.replace('</head>', `${portableLayoutCompatibility}</head>`);
writeFileSync(reportPath, html, "utf8");

const verification = await verifyPortableArtifact({
  actionTimeoutMs: 5000,
  artifactPath,
  htmlPath: reportPath,
  readyTimeoutMs: 15000,
  timeoutMs: 30000,
});
process.stdout.write(`${JSON.stringify({ ok: true, html: reportPath, verification })}\n`);
