# 2026-07-29 - Agent becomes an MCP client (DeepWiki first) - comparability boundary

The agent's toolbox now extends with tools from registered external MCP servers. This entry records
the first live run and, more importantly, the boundary it draws through the log: runs before and
after this change do not measure the same system.

## Setup

**Set** one live question, no eval set · **corpus** pre-variant era, post-ablation · **judge** none

A question about `langchain-ai/langchain`, absent from the corpus, through the agent pipeline with
one registered integration (DeepWiki).

## Result

The agent recognised the gap on hop 1, called `deepwiki__ask_question` on hop 2, and answered with
`mcp:deepwiki__ask_question` recorded as the source. Step one of the CRAG arc: the agent may ask
outside.

The measurement layer moved in the same change:

- `hit_at_k` and `MRR` rank corpus sources only; `mcp:*` evidence is excluded from retrieval metrics
  and stays in the log as provenance
- refusal semantics split: out-of-corpus questions with no remote evidence still expect a refusal
  (`refusal_accuracy`), while ones answered from remote evidence leave that pool and are counted as
  `answered_via_remote`
- every agent log snapshots the active integrations at run time (`metrics.config.mcp`)

## Decision

**From this change on, the agent has an external fallback. Runs before and after are not comparable
on refusal or retrieval axes without an explicit caveat**; `metrics.config.mcp` tells which world a
run belongs to.

## Caveats

- one question, no set, no judge: this is a boundary marker rather than a measurement
- what DeepWiki returns is not evaluated here at all; grounding in a tool answer says nothing about
  whether the tool was right
