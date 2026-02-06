import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type, type Static } from "@sinclair/typebox";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import OpenAI from "openai";

// Configuration
const INDEXED_PATHS = [
  { path: "/home/annapurna/sanctuary", label: "Sanctuary" },
  { path: "/home/annapurna/.openclaw/workspace/memory", label: "Session Logs" },
  { path: "/home/annapurna/.openclaw/workspace/MEMORY.md", label: "Core Memory" },
];

const CACHE_DIR = "/home/annapurna/.cache/memory-search";
const EMBEDDING_CACHE_FILE = path.join(CACHE_DIR, "embeddings.json");
const EMBEDDING_MODEL = "text-embedding-3-small";
const MAX_RESULTS = 10;
const MIN_SCORE = 0.1;
const BATCH_SIZE = 20; // Batch embedding requests

// Types
interface IndexedChunk {
  path: string;
  content: string;
  startLine: number;
  endLine: number;
  embedding?: number[];
  keywords: string[];
  contentHash: string;
}

interface SearchResult {
  path: string;
  content: string;
  score: number;
  startLine: number;
  endLine: number;
  matchType: "semantic" | "keyword" | "hybrid";
}

interface EmbeddingCache {
  [contentHash: string]: number[];
}

// Global state
let indexedChunks: IndexedChunk[] = [];
let openai: OpenAI | null = null;
let embeddingsAvailable = false;
let embeddingCache: EmbeddingCache = {};
let indexBuilt = false;

// Initialize OpenAI if API key is available
function initOpenAI(): boolean {
  const apiKey = process.env.OPENAI_API_KEY;
  if (apiKey) {
    openai = new OpenAI({ apiKey });
    embeddingsAvailable = true;
    return true;
  }
  return false;
}

// Load embedding cache from disk
function loadEmbeddingCache(): void {
  try {
    if (fs.existsSync(EMBEDDING_CACHE_FILE)) {
      const data = fs.readFileSync(EMBEDDING_CACHE_FILE, "utf-8");
      embeddingCache = JSON.parse(data);
    }
  } catch (e) {
    embeddingCache = {};
  }
}

// Save embedding cache to disk
function saveEmbeddingCache(): void {
  try {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(EMBEDDING_CACHE_FILE, JSON.stringify(embeddingCache));
  } catch (e) {
    console.error("Failed to save embedding cache:", e);
  }
}

// Hash content for caching
function hashContent(content: string): string {
  return crypto.createHash("md5").update(content).digest("hex");
}

// Tokenize text for keyword search
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

// Calculate BM25-style keyword score
function keywordScore(query: string[], docKeywords: string[]): number {
  const docSet = new Set(docKeywords);
  const matches = query.filter((q) => docSet.has(q));
  if (matches.length === 0) return 0;

  const tf = matches.length / Math.max(docKeywords.length, 1);
  const idf = Math.log(1 + Math.max(indexedChunks.length, 1) / (1 + matches.length));
  return tf * idf;
}

// Cosine similarity for embeddings
function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// Get embedding from OpenAI (single)
async function getEmbedding(text: string): Promise<number[] | null> {
  if (!openai || !embeddingsAvailable) return null;
  try {
    const response = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: text.slice(0, 8000),
    });
    return response.data[0].embedding;
  } catch (e) {
    console.error("Embedding error:", e);
    return null;
  }
}

// Get embeddings in batch from OpenAI
async function getEmbeddingsBatch(texts: string[]): Promise<(number[] | null)[]> {
  if (!openai || !embeddingsAvailable || texts.length === 0) {
    return texts.map(() => null);
  }
  try {
    const response = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: texts.map((t) => t.slice(0, 8000)),
    });
    return response.data.map((d) => d.embedding);
  } catch (e) {
    console.error("Batch embedding error:", e);
    return texts.map(() => null);
  }
}

// Split content into chunks
function chunkContent(content: string, chunkSize = 400): { text: string; startLine: number; endLine: number }[] {
  const lines = content.split("\n");
  const chunks: { text: string; startLine: number; endLine: number }[] = [];

  let currentChunk: string[] = [];
  let currentTokens = 0;
  let startLine = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineTokens = tokenize(line).length;

    if (currentTokens + lineTokens > chunkSize && currentChunk.length > 0) {
      chunks.push({
        text: currentChunk.join("\n"),
        startLine,
        endLine: i,
      });
      currentChunk = [];
      currentTokens = 0;
      startLine = i + 1;
    }

    currentChunk.push(line);
    currentTokens += lineTokens;
  }

  if (currentChunk.length > 0) {
    chunks.push({
      text: currentChunk.join("\n"),
      startLine,
      endLine: lines.length,
    });
  }

  return chunks;
}

// Collect all chunks from a directory (without embeddings yet)
function collectChunksFromDirectory(dirPath: string): IndexedChunk[] {
  const chunks: IndexedChunk[] = [];

  function walk(dir: string) {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.name.startsWith(".")) continue;
        if (entry.name === "node_modules") continue;

        if (entry.isDirectory()) {
          walk(fullPath);
        } else if (entry.isFile() && /\.(md|txt|json)$/i.test(entry.name)) {
          try {
            const content = fs.readFileSync(fullPath, "utf-8");
            const fileChunks = chunkContent(content);
            for (const chunk of fileChunks) {
              const contentHash = hashContent(chunk.text);
              chunks.push({
                path: fullPath,
                content: chunk.text,
                startLine: chunk.startLine,
                endLine: chunk.endLine,
                keywords: tokenize(chunk.text),
                contentHash,
                embedding: embeddingCache[contentHash], // Use cached if available
              });
            }
          } catch (e) {
            // Skip unreadable files
          }
        }
      }
    } catch (e) {
      // Skip inaccessible directories
    }
  }

  walk(dirPath);
  return chunks;
}

// Collect chunks from a single file
function collectChunksFromFile(filePath: string): IndexedChunk[] {
  try {
    const content = fs.readFileSync(filePath, "utf-8");
    const fileChunks = chunkContent(content);
    return fileChunks.map((chunk) => {
      const contentHash = hashContent(chunk.text);
      return {
        path: filePath,
        content: chunk.text,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        keywords: tokenize(chunk.text),
        contentHash,
        embedding: embeddingCache[contentHash],
      };
    });
  } catch (e) {
    return [];
  }
}

// Build the full index with embeddings
async function buildIndex(onProgress?: (msg: string) => void): Promise<number> {
  indexedChunks = [];
  loadEmbeddingCache();

  onProgress?.("Collecting files...");

  // Collect all chunks first
  for (const source of INDEXED_PATHS) {
    try {
      const stat = fs.statSync(source.path);
      if (stat.isDirectory()) {
        const chunks = collectChunksFromDirectory(source.path);
        indexedChunks.push(...chunks);
      } else if (stat.isFile()) {
        const chunks = collectChunksFromFile(source.path);
        indexedChunks.push(...chunks);
      }
    } catch (e) {
      // Path doesn't exist, skip
    }
  }

  onProgress?.(`Found ${indexedChunks.length} chunks`);

  // Now compute embeddings for chunks that don't have them
  if (embeddingsAvailable) {
    const chunksNeedingEmbeddings = indexedChunks.filter((c) => !c.embedding);

    if (chunksNeedingEmbeddings.length > 0) {
      onProgress?.(`Computing embeddings for ${chunksNeedingEmbeddings.length} chunks...`);

      // Process in batches
      for (let i = 0; i < chunksNeedingEmbeddings.length; i += BATCH_SIZE) {
        const batch = chunksNeedingEmbeddings.slice(i, i + BATCH_SIZE);
        const texts = batch.map((c) => c.content);
        const embeddings = await getEmbeddingsBatch(texts);

        for (let j = 0; j < batch.length; j++) {
          if (embeddings[j]) {
            batch[j].embedding = embeddings[j]!;
            embeddingCache[batch[j].contentHash] = embeddings[j]!;
          }
        }

        onProgress?.(`Embedded ${Math.min(i + BATCH_SIZE, chunksNeedingEmbeddings.length)}/${chunksNeedingEmbeddings.length}`);
      }

      // Save updated cache
      saveEmbeddingCache();
    } else {
      onProgress?.("All embeddings loaded from cache");
    }
  }

  indexBuilt = true;
  const withEmbeddings = indexedChunks.filter((c) => c.embedding).length;
  onProgress?.(`Index complete: ${indexedChunks.length} chunks, ${withEmbeddings} with embeddings`);

  return indexedChunks.length;
}

// Search the index
async function search(query: string, maxResults = MAX_RESULTS): Promise<SearchResult[]> {
  if (!indexBuilt || indexedChunks.length === 0) {
    await buildIndex();
  }

  const queryTokens = tokenize(query);
  const queryEmbedding = await getEmbedding(query);

  const scored: SearchResult[] = [];

  for (const chunk of indexedChunks) {
    let score = 0;
    let matchType: "semantic" | "keyword" | "hybrid" = "keyword";

    const kwScore = keywordScore(queryTokens, chunk.keywords);

    if (queryEmbedding && chunk.embedding) {
      const semScore = cosineSimilarity(queryEmbedding, chunk.embedding);
      matchType = kwScore > 0 ? "hybrid" : "semantic";
      score = 0.7 * semScore + 0.3 * kwScore;
    } else {
      score = kwScore;
      matchType = "keyword";
    }

    if (score >= MIN_SCORE) {
      scored.push({
        path: chunk.path,
        content: chunk.content,
        score,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        matchType,
      });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxResults);
}

// Tool parameter schema
const searchParamsSchema = Type.Object({
  query: Type.String({ description: "Search query - natural language or keywords" }),
  maxResults: Type.Optional(Type.Number({ description: "Maximum results to return (default 10)" })),
});

type SearchParams = Static<typeof searchParamsSchema>;

export default function (pi: ExtensionAPI) {
  // Initialize on load
  initOpenAI();
  loadEmbeddingCache();

  // Register the memory_search tool
  pi.registerTool({
    name: "memory_search",
    label: "Memory Search",
    description: `Search Annapurna's semantic memory including:
- Sanctuary (~/sanctuary/) - long-term knowledge base
- Session logs (~/.openclaw/workspace/memory/) - daily conversation logs
- Core memory (MEMORY.md) - identity and key information

Use for: recalling past conversations, finding decisions, looking up stored knowledge, checking what was discussed previously.`,
    parameters: searchParamsSchema,

    async execute(toolCallId, params: SearchParams, signal, onUpdate, ctx) {
      const { query, maxResults = MAX_RESULTS } = params;

      const progress = (msg: string) => {
        onUpdate?.({ content: [{ type: "text", text: msg }] });
      };

      progress("Searching memory...");

      // Ensure index is built
      if (!indexBuilt || indexedChunks.length === 0) {
        await buildIndex(progress);
      }

      const results = await search(query, maxResults);

      if (results.length === 0) {
        return {
          content: [{ type: "text", text: `No results found for: "${query}"` }],
          details: { query, results: [], indexedChunks: indexedChunks.length },
        };
      }

      // Format results
      const withEmbeddings = indexedChunks.filter((c) => c.embedding).length;
      let output = `Found ${results.length} results for: "${query}"\n`;
      output += embeddingsAvailable
        ? `(Semantic search enabled: ${withEmbeddings}/${indexedChunks.length} chunks have embeddings)\n\n`
        : "(Keyword search only - set OPENAI_API_KEY for semantic search)\n\n";

      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const relPath = r.path.replace("/home/annapurna/", "~/");
        output += `---\n### ${i + 1}. ${relPath}:${r.startLine}-${r.endLine} (${r.matchType}, score: ${r.score.toFixed(3)})\n\n`;
        output += r.content.trim() + "\n\n";
      }

      return {
        content: [{ type: "text", text: output }],
        details: {
          query,
          resultCount: results.length,
          indexedChunks: indexedChunks.length,
          embeddingsEnabled: embeddingsAvailable,
          matchTypes: results.map((r) => r.matchType),
        },
      };
    },
  });

  // Register /memory-reindex command
  pi.registerCommand("memory-reindex", {
    description: "Rebuild the memory search index",
    handler: async (args, ctx) => {
      ctx.ui.notify("Rebuilding memory index...", "info");
      indexBuilt = false;
      indexedChunks = [];
      embeddingCache = {}; // Clear cache to force re-embed
      const count = await buildIndex((msg) => ctx.ui.notify(msg, "info"));
      const withEmbeddings = indexedChunks.filter((c) => c.embedding).length;
      ctx.ui.notify(`Indexed ${count} chunks (${withEmbeddings} with embeddings)`, "success");
    },
  });

  // Register /memory-status command
  pi.registerCommand("memory-status", {
    description: "Show memory search status",
    handler: async (args, ctx) => {
      const withEmbeddings = indexedChunks.filter((c) => c.embedding).length;
      const status = [
        `Index built: ${indexBuilt ? "yes" : "no"}`,
        `Indexed chunks: ${indexedChunks.length}`,
        `Chunks with embeddings: ${withEmbeddings}`,
        `Embeddings available: ${embeddingsAvailable ? "yes (OPENAI_API_KEY set)" : "no"}`,
        `Cache file: ${EMBEDDING_CACHE_FILE}`,
        `Sources:`,
        ...INDEXED_PATHS.map((s) => `  - ${s.label}: ${s.path}`),
      ];
      ctx.ui.notify(status.join("\n"), "info");
    },
  });

  // Register /memory-clear-cache command
  pi.registerCommand("memory-clear-cache", {
    description: "Clear the embedding cache",
    handler: async (args, ctx) => {
      embeddingCache = {};
      try {
        if (fs.existsSync(EMBEDDING_CACHE_FILE)) {
          fs.unlinkSync(EMBEDDING_CACHE_FILE);
        }
        ctx.ui.notify("Embedding cache cleared", "success");
      } catch (e) {
        ctx.ui.notify("Failed to clear cache", "error");
      }
    },
  });
}
