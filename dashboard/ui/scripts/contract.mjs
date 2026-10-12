// Generate and check the dashboard contract JSON Schemas.
//
//   node scripts/contract.mjs generate   rewrite ../contract/*.schema.json from the .ts sources
//   node scripts/contract.mjs check      fail if a schema is stale or an example does not validate
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Ajv from "ajv";
import addFormats from "ajv-formats";
import { createGenerator } from "ts-json-schema-generator";

const contractDir = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "contract");
const SCHEMAS = [
  { source: "api-types.ts", schema: "api-v1.schema.json", examples: "api/" },
  { source: "ingest-types.ts", schema: "ingest-v1.schema.json", examples: "ingest/" },
];

function generate(source) {
  const schema = createGenerator({
    path: join(contractDir, source),
    type: "*",
    expose: "export",
    topRef: true,
    jsDoc: "extended",
    additionalProperties: false,
    sortProps: true,
    skipTypeCheck: true,
  }).createSchema("*");
  return JSON.stringify(sortKeys(schema), null, 2);
}

// Sort object keys recursively so the output is stable and diffs stay small.
function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, sortKeys(value[key])]),
    );
  }
  return value;
}

function validateExamples(entry, schemaText) {
  const index = JSON.parse(readFileSync(join(contractDir, "examples", "index.json"), "utf8"));
  const ajv = new Ajv({ strict: false, allErrors: true });
  addFormats(ajv);
  ajv.addSchema(JSON.parse(schemaText), "root");
  let failures = 0;
  for (const [file, def] of Object.entries(index)) {
    if (!file.startsWith(entry.examples)) continue;
    const data = JSON.parse(readFileSync(join(contractDir, "examples", file), "utf8"));
    const validate = ajv.getSchema(`root#/definitions/${def}`);
    if (!validate) {
      console.error(`examples/${file}: definition ${def} not found in ${entry.schema}`);
      failures += 1;
    } else if (!validate(data)) {
      console.error(`examples/${file} does not validate against ${def}:`);
      for (const err of validate.errors ?? []) console.error(`  ${err.instancePath} ${err.message}`);
      failures += 1;
    }
  }
  return failures;
}

const mode = process.argv[2];
if (mode !== "generate" && mode !== "check") {
  console.error("usage: contract.mjs generate|check");
  process.exit(2);
}
let failures = 0;
for (const entry of SCHEMAS) {
  const target = join(contractDir, entry.schema);
  const fresh = generate(entry.source);
  if (mode === "generate") {
    writeFileSync(target, fresh);
    console.log(`wrote ${entry.schema}`);
  } else if (readFileSync(target, "utf8") !== fresh) {
    console.error(`${entry.schema} is out of date; run npm run contract:generate`);
    failures += 1;
  }
  failures += validateExamples(entry, fresh);
}
if (failures) process.exit(1);
console.log(mode === "check" ? "contract ok" : "done");
