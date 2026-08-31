# What each role requires of a model

Four roles point at models, and each one imposes a requirement that is invisible in the model's
name and its benchmark scores. A model that fails its role's requirement does not degrade, it
fails, and usually it fails quietly enough to be mistaken for a bad result rather than a broken
run. This page says what each role needs and which incident put the line here.

The list of models we actually run, with sizes and licences, is not here on purpose: models come
and go, requirements do not.

## generation: tool calling, if the agent pipeline is to work at all

The agent sends `tools` on every call. Ollama decides whether a model accepts them from its
template, reports it as a capability, and refuses a request carrying `tools` with a 400 **before
inference** when the capability is missing. There is no partial support and no degraded mode.

```
ollama show <model>

  Capabilities
    completion
    tools        <- the agent pipeline needs this line
```

The failure is not loud where it matters. The first hop raises, the graph forces a final answer
with zero sources, and the model, given no context, writes an honest refusal. The run finishes,
every job reports done, and the report reads as a pipeline that answered nothing rather than as a
pipeline that never ran. On 30.08 that shape cost 400 questions across four control runs: 92 to 95
per cent refusals on a set whose questions all sit well inside the corpus.

`single_shot` does not send tools and does not need the capability. The trap is that one role
serves both pipelines: a generator chosen on single-shot numbers can be measured, accepted, and
merged while being unable to run the agent at all. Measure the role on both pipelines, or say in
the record that only one was measured and that the other is unknown.

## judging: not a reasoning-only model

The judge is asked for JSON against a schema at temperature 0, and its budget is `max_tokens`. A
model that reasons before answering spends that budget on the trace, and what arrives is a
truncated trace instead of a verdict.

Reasoning models come in three shapes, and only the middle one is safe to hand this role:

- **no reasoning**: answers directly, nothing to switch off;
- **hybrid**: reasons by default, and the reasoning is switchable (`think: false` on the native
  API, `reasoning_effort: "none"` on the OpenAI-compatible one);
- **reasoning only**: the chat template inserts the opening think tag itself, and there is no
  switch. Model cards say so in one line ("supports only thinking mode"), which is a line worth
  reading before a grid rather than after it.

Two more things make this sharper than it looks. A tag in a registry can point at a reasoning-only
build while the documentation a search engine offers describes the switchable sibling under the
same family name: check the digest, not the name. And a client that passes only `temperature`,
`max_tokens` and `response_format` cannot switch reasoning off even for a hybrid, because the knob
travels in neither of those fields.

## embedding: the corpus is committed to it

The embedding model is the one role whose replacement is not a config change. Its output dimension
is the width of a database column and of every vector index; its numbers are baked into every
stored chunk and every stored question. Changing it means re-embedding the whole corpus and every
question set, and until both are done, distances measured before and after are not comparable and
no threshold calibrated on the old ones holds.

Requirement: the dimension the schema was built for, and the languages the corpus is actually in.
Everything else is a re-index, which is a measured operation, not a swap.

## paraphrasing: fluency in the target language, nothing else

This role builds question sets rather than serving requests. It needs no tools and its latency does
not matter. What it does need is real fluency in the language it writes: a set built by a model
that writes stilted Russian measures the model's Russian, not the corpus.

It has one operational hazard that has nothing to do with quality. A model loaded for a one-off
data-preparation job and left resident takes the card away from whatever runs next. On 30.08 the
paraphrasing model held the card, the reranker fell back to the CPU with a warning rather than an
error, and the run would have been an order of magnitude slower while reporting nothing unusual. A
role raised for a task releases the card after it, including on the failure path.

## The reranker is not a role

The cross-encoder is not in ollama, has no entry among the roles, and is loaded in-process through
sentence-transformers. Two consequences follow, and both have bitten.

Residency checks that ask ollama what is on the card cannot see it, so any statement about how much
card is free is an estimate unless the reranker was measured on its own. And when the card is full,
it does not fail: `RERANK_DEVICE=auto` falls back to CPU with a warning, and the same run then
takes roughly thirty times longer per question while producing identical output. The guard for this
is a check that asks the reranker where it actually is, after it has loaded, and refuses the run
rather than letting it proceed slowly.

## The card decides more than quality does

Every role that points at an ollama model, plus the reranker, competes for one card. Residency is
capped (`OLLAMA_MAX_LOADED_MODELS`) and resident models are kept alive, so the third arrival evicts
the first. This is why a role change is an arithmetic problem before it is a quality problem: the
model, the embedder, and the reranker if it is on, against the card.

The choice that follows from that arithmetic is a decision, not a fact, and it belongs in the
record with its price attached. Reranking is off by default here because the generator the agent
needs does not leave room for it, and reranking is worth +0.0454 [+0.0250, +0.0670] of section MRR
on the criterion set. Both halves of that sentence are true at once.

## Where the role's model name actually lives

`config.yaml` carries the inference options for each role, and the model name in it is what
bootstrap assigns on an empty database. Once a role is assigned, the name is served from the
database and bootstrap leaves it alone, because the door for changing it is `PUT /v1/role` and a
restart must not undo a deliberate change.

The consequence is that editing the file changes nothing about which model answers. Run snapshots
record what the database served, so runs stay honest and only the file lies. Change the role
through the route, and read the file as a declaration rather than as the current state.

## Before giving a role to a model

1. `ollama show <tag>`: capabilities, context, and the sampling parameters baked into the template.
   Check the digest when a family ships several builds under one name.
2. Add up the card: the model, the embedder, the reranker if it is on, against what the card has.
3. Smoke ten questions and read the distribution of outcomes, not just that the job finished. A
   dead pipeline shows up as every question landing in one bucket.
4. Change the role through `PUT /v1/role`, then run the preflight.
