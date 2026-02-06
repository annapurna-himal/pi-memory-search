# pi-memory-search

A semantic memory search extension for [Pi](https://github.com/badlogic/pi-mono) coding agent.

Indexes Markdown files and session logs, generates OpenAI embeddings, and provides a `memory_search` tool the LLM can call to recall past conversations, decisions, and stored knowledge.

## Features

- **Semantic search** via OpenAI `text-embedding-3-small` embeddings
- **Configurable search paths** — index any directories of Markdown files
- **Persistent vector store** — embeddings cached to disk, incremental updates
- **Content-aware chunking** — splits by headings, respects document structure
- **Session log support** — search daily conversation logs alongside knowledge bases

## Installation

```bash
pi install git:github.com/annapurna-himal/pi-memory-search
```

Or clone and symlink for development:

```bash
git clone git@github.com:annapurna-himal/pi-memory-search.git ~/repos/pi-memory-search
ln -s ~/repos/pi-memory-search ~/.pi/agent/extensions/memory-search
cd ~/repos/pi-memory-search && npm install
```

## Configuration

Requires an `OPENAI_API_KEY` environment variable for embeddings.

The extension registers a `memory_search` tool that the LLM can call with a natural language query. Configure search paths by modifying the indexed directories in `index.ts`.

## Usage

Once installed, the LLM can search your knowledge base:

```
memory_search("version control strategy")
memory_search("what did we decide about the API design")
```

## License

MIT
