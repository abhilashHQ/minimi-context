# minimi-context

A person context layer on top of minimi. It organizes the memories minimi already generates around people, then retrieves what matters at the moment you return to someone or ask.

Implementation contract: [SPEC.md](SPEC.md).

## Screen Memory Engine: P1 search-only baseline

This is the go or no-go test that runs before the engine is built. A plain search-only baseline answers the fixed questions in `corpus/expected.yaml` over a fictional corpus (`corpus/scenario_a.json`, `corpus/scenario_b.json`). A human judges the saved transcripts. All names and messages are fictional.

### Install (Windows, PowerShell, Python 3.11)

```powershell
cd C:\path\to\minimi-context
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
pytest
ruff check .
```

### Build scenario A and B

The build makes one Gemini call per conversation and embeds each memory with `all-MiniLM-L6-v2`. The first run downloads that model to the Hugging Face cache, and the server reuses it.

```powershell
$env:GEMINI_API_KEY = "<your key>"
python -m baseline.build --scenario a --db runs\p1\a.db
python -m baseline.build --scenario b --db runs\p1\b.db
```

Add `--rebuild` to delete an existing database and build it again. Without it, the build refuses to append memories to a database that already has them.

### Claude Desktop config

Edit `%APPDATA%\Claude\claude_desktop_config.json`. Use one scenario at a time and restart Claude Desktop after switching. Replace `C:\\path\\to\\minimi-context` with the real path.

Scenario A:

```json
{
  "mcpServers": {
    "memory": {
      "command": "C:\\path\\to\\minimi-context\\.venv\\Scripts\\python.exe",
      "args": [
        "-m", "baseline.server",
        "--db", "C:\\path\\to\\minimi-context\\runs\\p1\\a.db",
        "--now", "2026-11-12T20:00:00+05:30"
      ]
    }
  }
}
```

Scenario B:

```json
{
  "mcpServers": {
    "memory": {
      "command": "C:\\path\\to\\minimi-context\\.venv\\Scripts\\python.exe",
      "args": [
        "-m", "baseline.server",
        "--db", "C:\\path\\to\\minimi-context\\runs\\p1\\b.db",
        "--now", "2026-11-13T17:00:00+05:30"
      ]
    }
  }
}
```

The server exposes exactly three tools: `search_memory`, `list_active_threads` and `get_latest_context`. Every response includes `as_of`, set to `--now`.

### Claude Desktop project instructions

Create one Claude Desktop project for P1 and use it for every P1 chat. Set its instructions to exactly:

```text
Use the memory tools to answer. If memory has no answer, say you don't know.
```

### Running the questions and saving transcripts

1. Load the scenario's config (Q1 to Q6 use scenario A, Q7 uses scenario B) and restart Claude Desktop.
2. Start a new chat in the P1 project for each question and paste the question exactly as written in `corpus/expected.yaml`.
3. When the answer finishes, expand every tool call, then copy the question, each tool call with its result, and the final answer.
4. Save that into `runs\p1\<scenario>-<question id>.md`, for example `runs\p1\a-q1.md` and `runs\p1\b-q7.md`. Put the date and the model name on the first line.
5. Judge each transcript by hand against the required elements in `corpus/expected.yaml`.

`runs/` is git-ignored, so databases and transcripts stay local.
