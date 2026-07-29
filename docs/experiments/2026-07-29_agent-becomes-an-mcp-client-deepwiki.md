# 2026-07-29 - Agent becomes an MCP client (DeepWiki first) - comparability boundary

The agent's toolbox now extends with tools from registered external MCP servers (see README, MCP
client section). First live run: a question about `langchain-ai/langchain` (absent from the corpus) -
the agent recognized the gap on hop 1, called `deepwiki__ask_question` on hop 2, answered with
`mcp:deepwiki__ask_question` as the recorded source. The CRAG arc, step 1: the agent MAY ask outside.

Measurement layer was updated in the same change (external audit, fix-loop in the branch):
- `hit_at_k`/`MRR` rank only corpus sources - `mcp:*` evidence is excluded from retrieval metrics
  (it stays in the log as provenance).
- refusal semantics split: out-of-corpus questions with no remote evidence still expect a refusal
  (`refusal_accuracy`); ones answered from remote evidence are excluded from that pool and counted
  separately as `answered_via_remote`.
- every agent log now snapshots the active integrations at run time (`metrics.config.mcp`).

**Comparability boundary: from this change on, the agent has an external fallback. Runs before and
after are not comparable on refusal or retrieval axes without an explicit caveat; check
`metrics.config.mcp` to tell which world a run belongs to.**
