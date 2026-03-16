# gdf — Distributed Federated LLM Training

A volunteer GPU compute network. People contribute their GPUs to collectively train language models, and anyone can chat with the results.

Two commands define the experience:

```
gdf contribute        # pick a model, your GPU auto-trains
gdf chat              # talk to models, router picks the best specialist
```

## Try it locally

Run the full stack (hub + contribute + chat) on your own machine:

```bash
# Terminal 1 — start the hub
gdf init --name hub_model.pt
gdf hub --model hub_model.pt --token local-dev

# Terminal 2 — contribute your GPU
gdf contribute

# Terminal 3 — chat with the model
gdf chat
```

Or use the launcher script: `./scripts/run_hub.sh` starts the hub in one command.

### Deploy to a server

To move from local testing to a real deployment:

1. Run the hub on a server with a public IP
2. Edit `models.json` — change `hub_url` to your server's address and set a real `token`
3. Share `gdf contribute` with volunteers

## Install

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch. Works on CPU, CUDA, and Apple Silicon (MPS).

## Quick start

### Chat with models

```bash
gdf chat                     # auto-routes each question to the best specialist
gdf chat general-7b          # chat with a specific model
```

Inside the chat REPL:

| Command | What it does |
|---|---|
| `/models` | List available specialists and network models |
| `/switch <name>` | Pin conversation to a specific model |
| `/auto` | Switch back to auto-routing |
| `/quit` | Exit |

The router is keyword-based (not an ML model) — it scores each specialist by keyword overlap, domain match, and quality, then picks the best one for your question.

### Contribute your GPU

```bash
gdf contribute               # pick from available models
gdf contribute general-7b    # start training a specific model
```

This connects to a hub, pulls the latest model, trains on Wikipedia articles, and pushes your improvements back. Runs until you hit Ctrl+C.

### Train your own model locally

```bash
gdf init                          # create a new model
gdf learn https://example.com     # learn from a URL
gdf learn mybook.txt              # learn from a file
gdf crawl ~/Documents             # learn from all files in a folder
gdf autolearn                     # self-learn from Wikipedia autonomously
gdf generate "Once upon a time"   # generate text
```

### Specialists

Instead of one giant general model, gdf supports a network of domain-specific specialists. Each is deeply trained on a specific topic.

```bash
gdf specialist create cardiology -d medical -k heart -k cardiac
gdf specialist train cardiology heart_papers.txt
gdf specialist autolearn cardiology     # auto-learns from Wikipedia using keywords
gdf specialist ask "What causes atrial fibrillation?"
gdf specialist list
```

When you use `gdf chat` without specifying a model, the router automatically picks the best specialist for each question.

## Architecture

### How the network works

```
                    +-----------+
                    |  Registry |  (models.json on GitHub)
                    +-----+-----+
                          |
              lists available models
                          |
         +----------------+----------------+
         |                                 |
    gdf contribute                    gdf chat
    (picks a model,                   (picks a model,
     connects to hub)                  downloads, talks)
         |                                 |
    +----v----+                       +----v----+
    |   Hub   |  <-- merges deltas    |  Local  |
    +---------+      from peers       |  Model  |
    /  |   |  \                       +---------+
   /   |   |   \
  P1  P2  P3  P4   (peers train and push)
```

**Hub** — coordination server that collects contributions and merges them. Anyone can run one with `gdf hub`.

**Peer** — a volunteer GPU. Pulls the latest model, trains on web data, pushes only what changed (delta compression).

**Registry** — a JSON file on GitHub listing available models, their hub URLs, and public tokens. Cached locally for 1 hour.

### Delta compression

Peers don't send the full model. They compute `current_weights - base_weights`, zero out tiny changes, convert to fp16, and compress with zlib. For a 7B model where 5% of weights changed, this reduces transfer from ~14 GB to ~200-400 MB.

### Delta validation

The hub rejects poisoned contributions:

- **NaN/Inf check** — any tensor containing NaN or Inf is rejected
- **Magnitude check** — any weight change larger than 10.0 is rejected

This prevents both accidental corruption and intentional model poisoning.

### P2P model distribution

Models are split into chunks (one per layer), described by a manifest with SHA-256 hashes. When a new peer joins:

1. Fetches the manifest from the hub (~1 KB)
2. Asks the tracker which peers have which chunks
3. Downloads chunks from peers first, hub as fallback
4. Verifies each chunk's hash
5. Reassembles the model
6. Starts seeding chunks to other peers

```
Hub bandwidth comparison for a 7B model with 10k peers:
  Centralized:  14 GB x 10,000 = 140 TB from hub
  P2P:          14 GB x ~10    = 140 GB from hub (seed to first 10 peers)
```

### Hierarchical merging

Hubs can form a tree for large-scale training:

```
        Root Hub
       /    |    \
    Hub1  Hub2  Hub3    (regional hubs, ~100 peers each)
    /|\   /|\   /|\
    ...   ...   ...
```

Each hub merges small groups locally, then pushes upstream. This preserves more knowledge than merging 10k models at once.

## Running a hub

```bash
gdf init                      # create the base model
gdf hub --port 7677           # start the hub
```

The hub prints a token and a command to share with peers:

```
  TOKEN: abc123...
  Share this command with peers:
    gdf peer http://YOUR_IP:7677 --token abc123...
```

Hub options:

| Flag | Default | Description |
|---|---|---|
| `--port` | 7677 | Port to listen on |
| `--merge-every` | 3 | Merge after N contributions |
| `--strategy` | fedavg | Merge strategy (fedavg, ties) |
| `--token` | auto | Auth token |
| `--parent` | none | Parent hub URL (for hierarchy) |

## Model registry

The file `models.json` in the repo root lists available models:

```json
[
  {
    "name": "general-7b",
    "description": "General-purpose 7B language model",
    "hub_url": "http://hub1.gdf.network:7677",
    "token": "public-token-here",
    "size": "14GB",
    "status": "active"
  }
]
```

Tokens are public by design — security comes from delta validation, not access control.

`gdf contribute` and `gdf chat` both read from this registry. It's cached locally at `~/.gdf/registry.json` and refreshed every hour.

## Project structure

```
src/gdf/
  cli.py            CLI commands (chat, contribute, hub, peer, learn, etc.)
  api.py            GDFModel — high-level create/load/train/generate/merge
  model.py          TinyTransformer architecture
  trainer.py        OnlineTrainer with replay buffer
  selflearn.py      Autonomous Wikipedia learning loop
  distributed.py    Hub, Peer, delta compression, P2P wiring
  torrent.py        Chunk splitting, seeding, tracking, downloading
  registry.py       Fetch model registry from GitHub
  specialists.py    Specialist registry and keyword router
  merging.py        Model merging strategies (fedavg, ties, task arithmetic)
  growth.py         Grow models to larger architectures
  serialization.py  Save/load models
  tokenizer.py      Byte-level tokenizer
  bpe.py            BPE tokenizer training
  fetcher.py        URL fetching and text extraction
  crawler.py        File discovery and bulk training
  config.py         Config management
  device.py         GPU/CPU/MPS detection

tests/
  test_torrent.py   P2P chunk distribution tests
  test_security.py  Delta validation tests
  test_registry.py  Model registry tests
  test_model.py     Transformer architecture tests
  test_trainer.py   Training loop tests
  test_merging.py   Merge strategy tests

models.json         Model registry (repo root)
```

## All commands

| Command | Description |
|---|---|
| `gdf chat` | Chat with models (auto-routed or specific) |
| `gdf contribute` | Contribute GPU to train a network model |
| `gdf init` | Create a new model |
| `gdf learn <source>` | Learn from a URL or file |
| `gdf crawl [folder]` | Learn from all files in a folder |
| `gdf autolearn` | Self-learn from Wikipedia |
| `gdf generate [prompt]` | Generate text |
| `gdf status` | Show model info |
| `gdf grow` | Grow model to larger architecture |
| `gdf merge` | Merge multiple models |
| `gdf hub` | Run a coordination hub |
| `gdf peer <url>` | Join as a training peer |
| `gdf specialist create` | Create a domain specialist |
| `gdf specialist train` | Train a specialist |
| `gdf specialist ask` | Ask a question (auto-routed) |
| `gdf specialist list` | List specialists |
| `gdf specialist autolearn` | Auto-learn for a specialist |
| `gdf specialist domains` | Show suggested domains |
