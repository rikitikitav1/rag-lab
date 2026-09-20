import statistics
import sys

import limits
import outcomes
import token_fields
from engines import DANGLING, answer_parsers, engine_of, registered_names
from engines.card import NO_RESIDENCY
from evals.loaders import load_logs
from evals.pools import (
    ALL_OUTCOMES,
    JOINS_BOTH_JUDGES,
    POOLS,
    Ambiguous,
    by_question,
    has_remote_evidence,
    joins_both_judges,
    outcome,
    split,
)
from evals.stats import delta_stats, deltas_over, mean_of, tally
from use_cases import rejudge
from use_cases.agent_policy import FallbackReason

OUTCOMES = ALL_OUTCOMES
AXES = rejudge.AXES
GATE_REASONS = ("empty", "weak", "off_topic")


# both doors onto `compare` walk every run named, so the caps belong here and not at one door
def named_runs(run_names: list[str]) -> list[str]:
    named = list(dict.fromkeys(run_names))
    if not named:
        raise ValueError("run_names must not be empty")
    if len(named) > limits.MAX_RUNS:
        raise ValueError(f"{len(named)} runs is over the cap of {limits.MAX_RUNS}")
    too_long = [n for n in named if len(n) > limits.MAX_RUN_NAME]
    if too_long:
        raise ValueError(f"run names over {limits.MAX_RUN_NAME} characters: {too_long[:3]}")
    return named


def summarize(logs) -> dict:
    marks = [outcome(ql) for ql in logs]
    latency = [ql.elapsed for ql in logs if ql.elapsed is not None]
    remote = [ql for ql in logs if has_remote_evidence(ql)]
    home = [
        ql
        for ql, mark in zip(logs, marks, strict=True)
        if mark == "answered" and not has_remote_evidence(ql)
    ]
    # an open gate that answered from the corpus anyway is a failed handoff, not a shut gate
    opened = [ql for ql in home if (ql.metrics or {}).get("fallback_opened")]
    reasons = [(ql.metrics or {}).get("fallback_reason") for ql in logs]
    return {
        "n": len(logs),
        "judged": sum(1 for ql in logs if ql.faithfulness is not None),
        "faithfulness": mean_of(ql.faithfulness for ql in logs),
        "relevance": mean_of(ql.relevance for ql in logs),
        "completeness": mean_of(ql.completeness for ql in logs),
        "answered_via_remote": len(remote),
        "answered_from_corpus": len(home),
        "answered_from_corpus_gate_shut": len(home) - len(opened),
        "answered_from_corpus_opened_no_evidence": len(opened),
        "answered_from_corpus_rate": round(len(home) / len(logs), 3) if logs else None,
        "gate_fired": sum(1 for r in reasons if r in GATE_REASONS),
        # its own count: a grader that emptied the context is not the gate firing
        "graded_out": sum(1 for r in reasons if r == FallbackReason.graded_out),
        "latency_avg": mean_of(latency, digits=1),
        "latency_p50": round(statistics.median(latency), 1) if latency else None,
        "outcomes": {o: marks.count(o) for o in OUTCOMES},
    }


def _client(logs) -> str | None:
    for ql in logs:
        orchestrator = ((ql.metrics or {}).get("config") or {}).get("orchestrator") or {}
        if orchestrator.get("client"):
            return orchestrator["client"]
    return None


def paired(left, right, axis) -> dict:
    def scores(logs):
        return {
            question: None if getattr(ql, axis) is None else float(getattr(ql, axis))
            for question, ql in by_question(logs).items()
        }

    before, after = scores(left), scores(right)
    # in the order `right` arrives, which is the order the bootstrap was drawn over
    ids = [ql.question_id for ql in right if ql.question_id in before]
    kept = [i for i in ids if before[i] is not None and after[i] is not None]
    deltas = deltas_over(before, after, ids)

    counted = tally(deltas)
    result = {
        "n": len(deltas),
        "left": mean_of(before[i] for i in kept),
        "right": mean_of(after[i] for i in kept),
        "better": counted["better"],
        "worse": counted["worse"],
        "mean_delta": None,
        "ci95": None,
        "p": None,
    }
    if deltas:
        stats = delta_stats(deltas)
        result["mean_delta"] = stats["mean_delta"]
        result["ci95"] = stats["ci95"]
        # `wilcoxon_p` already answers the all-zero case with 1.0, and null cannot enter Holm
        result["p"] = stats["p"]
    return result


def _judged(ql, axis) -> dict:
    return (ql.metrics or {}).get(axis) or {}


# a mean can hold still while the ruler moves: 1 to 0 on one row and 0 to 1 on another cancel
def verdicts(left: list, right: list) -> dict:
    before, after = by_question(left), by_question(right)
    shared = [q for q in after if q in before]
    axes = {}
    for axis in AXES:
        both = [(before[q], after[q]) for q in shared
                if getattr(before[q], axis) is not None and getattr(after[q], axis) is not None]
        tokens = [(_judged(a, axis).get(token_fields.JUDGE_PROMPT),
                   _judged(b, axis).get(token_fields.JUDGE_PROMPT)) for a, b in both]
        pair = paired(left, right, axis)
        axes[axis] = {
            "comparable": len(both),
            "disagree": sum(1 for a, b in both if float(getattr(a, axis)) != float(getattr(b, axis))),
            # scored on one side only: neither a match nor a clash, and left out of both counts
            "one_sided": sum(1 for q in shared
                             if (getattr(before[q], axis) is None) != (getattr(after[q], axis) is None)),
            # the means `paired` reports for this axis, over the same rows scored on both sides
            "left": pair["left"],
            "right": pair["right"],
            # the judge read fewer tokens on one side: its context was cut, or the tokenizer differs
            "prompt_tokens_differ": sum(1 for a, b in tokens
                                        if a is not None and b is not None and a != b),
            "seconds_left": mean_of((_judged(a, axis).get("elapsed") for a, _ in both), digits=1),
            "seconds_right": mean_of((_judged(b, axis).get("elapsed") for _, b in both), digits=1),
        }
    comparable = sum(a["comparable"] for a in axes.values())
    disagree = sum(a["disagree"] for a in axes.values())
    return {
        "questions": len(shared),
        "comparable": comparable,
        "disagree": disagree,
        "disagree_rate": round(disagree / comparable, 3) if comparable else None,
        "axes": axes,
    }


# the report's shape, raised with every field it gains
SCHEMA = 16


class TwoJudges(Ambiguous):
    pass


# the stamp exists so a reader can ask "did these arms share one code"; without this nobody ever asked
def code_by_run(runs: dict[str, list]) -> dict:
    said = {}
    for name, logs in runs.items():
        seen = {(ql.metrics or {}).get("config", {}).get("code_version") for ql in logs}
        said[name] = sorted(v for v in seen if v)
    every = {v for versions in said.values() for v in versions}
    return {"by_run": said, "one_code": len(every) <= 1,
            "reads": "the code each run's rows were written by; two arms on two stamps compare two trees"}


def compare(runs: dict[str, list]) -> dict:
    residency = residencies(runs)
    # two judges are two rulers: a difference of their means measures nothing
    if residency["one_engine_name"] is False:
        raise TwoJudges(
            f"{residency['read_this_first']}: judged on {residency['engine_names_by_run']}."
            " Read each arm alone through `run_metrics`, or rejudge one arm on the other's engine"
        )
    by_pool = {name: split(logs) for name, logs in runs.items()}
    names = list(runs)

    pools = {}
    for pool in POOLS:
        if pool == "rejected" or not any(by_pool[name][pool] for name in names):
            continue
        pairs = [
            {
                "left": left,
                "right": right,
                # a pair that also swaps the model client measures two changes, not one
                "isolates_orchestrator": _client(by_pool[left][pool])
                == _client(by_pool[right][pool]),
                **{
                    axis: paired(by_pool[left][pool], by_pool[right][pool], axis)
                    for axis in AXES
                },
            }
            for i, left in enumerate(names)
            for right in names[i + 1 :]
        ]
        pools[pool] = {
            "arms": {name: summarize(by_pool[name][pool]) for name in names},
            "pairs": pairs,
        }

    scored = {
        name: [ql for pool, logs in by_pool[name].items() if pool != "rejected" for ql in logs]
        for name in names
    }
    # pools differ in what they should do, so their blend ranks nothing: kept for latency only
    return {
        "schema": SCHEMA,
        "runs": names,
        "outcome_rule": outcomes.RULE,
        "residency": residency,
        # the treatment, not a fault: two generators on two engines is what a pair of arms compares
        "answering_engines_by_run": {name: _answering_engines(logs) for name, logs in runs.items()},
        "code": code_by_run(runs),
        # two servers apply their own penalty unasked (1.05 against 1.1): aligned first, then compared
        **_answering_penalties_of(runs),
        # the correlation's own predicate, called not restated: one label stood over two selections
        "correlation_population": {
            "predicate": JOINS_BOTH_JUDGES,
            "n": {name: sum(1 for ql in logs if joins_both_judges(ql))
                  for name, logs in runs.items()},
        },
        "pools": pools,
        "blended_do_not_rank": {name: summarize(logs) for name, logs in scored.items()},
        # a pair only, over the rows the rest of the report reads: another population, other shares
        "verdicts": verdicts(*scored.values()) if len(scored) == 2 else None,
    }


# in disqualifying order: a backend, then the arms themselves, then the ruler, then the reload
def _what_to_read_first(
    one_engine: bool | None,
    one_prompt: bool | None,
    one_residency: bool | None,
    one_engine_name: bool | None,
    one_deterministic: bool | None,
    one_parser: bool | None = None,
    remote_judge: bool = False,
    one_sampler: bool | None = None,
    one_budget: bool | None = None,
    cut: int = 0,
    one_grammar: bool | None = None,
) -> str | None:
    if one_engine_name is False:
        return (
            "arms judged on differently named engines are not comparable, and the address cannot "
            "show it: two engines take the same host and port in turn"
        )
    if one_engine is False:
        return (
            "arms judged on different engines are not comparable: batching, kernels and "
            "quantisation all differ, and residency does not even mean the same thing on both"
        )
    if one_deterministic is False:
        return (
            "the arms retrieved different sources or ranked them differently on the same questions, "
            "so they are not one arm read twice: unless retrieval is the treatment, this contrast "
            "measures a pipeline that changed underneath it"
        )
    if one_prompt is False:
        return (
            "arms scored by different judge prompt versions are two rulers, not one instrument "
            "read twice: unless the prompt is the treatment, this contrast measures the prompt"
        )
    if one_parser is False:
        return (
            "arms whose judge answers were cut by different parsers read different texts: the cut "
            "shapes what the score is read from, so this contrast measures the parser"
        )
    if one_grammar is False:
        return (
            "arms whose judge decoded its JSON by different rules wrote different replies: with free "
            "whitespace one reply wrote tabs until its limit, and without it the same row got another "
            "reason and a score, so this contrast measures the grammar"
        )
    if one_sampler is False:
        return (
            "arms whose judge sampled by a different temperature or seed chose their tokens by "
            "different rules, so this contrast measures the sampler"
        )
    if one_budget is False and cut:
        return (
            f"arms whose judge had different output budgets differ where a budget cut: {cut} verdicts "
            "ended on the limit, and there this contrast measures the budget"
        )
    if one_residency is False:
        return (
            "arms judged across a reload are not comparable directly: measured on ollama, the same "
            "judge moves 14% of its scores and 58% of its reason texts on byte-identical input, and "
            "a pair whose own floor was never measured cannot borrow that one"
        )
    if one_residency is None and remote_judge:
        return (
            "a remote judge has no residency: two passes are one instrument only while the broker "
            "keeps serving the same weights, and nothing the stand reads can tell"
        )
    if one_residency is None:
        return "rows judged before this was recorded carry no residency, so nothing can be said"
    if one_engine_name is None:
        return (
            "at least one arm recorded no engine name, so which engine stood behind it cannot be "
            "said, and the address alone does not tell two engines apart"
        )
    if one_engine is None:
        return (
            "the residency matches, but at least one arm recorded no engine, so whether both ran "
            "on one backend cannot be said, and that outranks any reading of the residency"
        )
    if one_prompt is None:
        return (
            "rows judged before the prompt version reached the row carry none, so whether both "
            "arms were scored by one ruler cannot be said"
        )
    if one_deterministic is None:
        return (
            "no question was retrieved by both arms with its sources recorded, so whether the "
            "deterministic half of the pipeline held still cannot be said"
        )
    if one_budget is False:
        return (
            "the arms' judge had different output budgets and no verdict reached either limit, so the "
            "budget changed nothing here; the rest of the reading holds as for one residency"
        )
    return (
        "one residency is necessary, not sufficient: two arms with identical rows, order and "
        "prompt still differed on 4 of 50 rows, so this contrast measures its own floor rather "
        "than inheriting a zero. A reading that rests on the reason text holds only here, and "
        "`seed` at temperature zero says the sampler took no part, not that a pass repeats"
    )


# a stamp from before the server's JSON rules were read carries neither key, and reads as its own rule
def _grammar_of(added: dict) -> str:
    return f"{added.get('json_backend')}:{added.get('json_disable_any_whitespace')}"


# per axis, because three axes carry three versions and their union is three by construction
def _one_ruler(by_run: dict) -> bool | None:
    # silence is not a match here either: one arm scored on one axis cannot agree with three
    seen = {frozenset(versions) for versions in by_run.values()}
    shared = set.intersection(*(set(v) for v in by_run.values())) if by_run else set()
    if not by_run or not shared or len(seen) != 1:
        return None
    for axis in shared:
        seen = {v for versions in by_run.values() for v in versions[axis]}
        if len(seen) != 1:
            return False
    return True


# None where any arm is silent: an empty set used to drop out and read as agreement
def _all_agree(by_run: dict) -> bool | None:
    if not by_run or any(not seen for seen in by_run.values()):
        return None
    return len({one for seen in by_run.values() for one in seen}) == 1


# the floats beside the ranks are not read: a distance that moved is the embedder's own story
def _footprint(ql) -> tuple | None:
    seen = ql.sources
    if not seen:
        return None
    return tuple(
        (one.get("source"), one.get("hop"), one.get("vector_rank"), one.get("keyword_rank"))
        for one in seen
    )


# the deterministic half must be equal, not close: two arms that retrieved differently are two arms
def _one_retrieval(runs: dict[str, list]) -> bool | None:
    by_run = {name: by_question(logs) for name, logs in runs.items()}
    if len(by_run) < 2:
        return None
    shared = set.intersection(*(set(seen) for seen in by_run.values()))
    compared = 0
    for question in shared:
        prints = [_footprint(by_run[name][question]) for name in by_run]
        # a row that recorded no source says nothing about the pipeline, and silence is not a clash
        if any(one is None for one in prints):
            continue
        compared += 1
        if len(set(prints)) != 1:
            return False
    return True if compared else None


# a single-shot row with no context answered NO_RESULTS itself, and older rows named a generator anyway
def _asked_the_generator(ql) -> bool:
    if "generation" in (getattr(ql, "models", None) or {}) and ql.models["generation"] is None:
        return False
    return not (getattr(ql, "pipeline", None) == "single_shot"
                and (getattr(ql, "answer", None) or "").strip() == outcomes.NO_RESULTS)


# read off the run snapshot's `config.engines`, per role; a row older than that key names nothing
def _answering_engines(logs: list) -> dict[str, list[str]]:
    seen: dict[str, set] = {}
    for ql in logs:
        asked = _asked_the_generator(ql)
        for role, engine in (((ql.metrics or {}).get("config") or {}).get("engines") or {}).items():
            if str(role) == "generation" and not asked:
                continue
            seen.setdefault(str(role), set()).add(engine)
    return {role: sorted(engines) for role, engines in sorted(seen.items())}


# the penalty each answering role ran with: sent where the call named it, else what its engine applied on its own
def _answering_penalties(logs: list) -> dict[str, list]:
    seen: dict[str, set] = {}
    for ql in logs:
        if not _asked_the_generator(ql):
            continue
        cfg = (ql.metrics or {}).get("config") or {}
        sent, added = cfg.get("samplers") or {}, cfg.get("engine_added") or {}
        for role in (set(sent) | set(added)) - {"embedding", "reranking"}:
            value = (sent.get(role) or {}).get("repetition_penalty", (added.get(role) or {}).get("repetition_penalty"))
            seen.setdefault(str(role), set()).add(value)
    return {role: sorted(values, key=str) for role, values in sorted(seen.items())}


# a row from before the penalty was stamped reads as unknown, and unknown matches nothing
def _answering_penalties_of(runs: dict[str, list]) -> dict:
    by_run = {name: _answering_penalties(logs) for name, logs in runs.items()}
    held = {value for roles in by_run.values() for value in roles.get("generation") or [None]}
    one = None if len(by_run) < 2 or None in held or "unknown" in held else len(held) == 1
    return {"answering_penalties_by_run": by_run, "one_answering_penalty": one}


# two arms judged across a reload are two instruments: 14% of scores move on identical input
def residencies(runs: dict[str, list]) -> dict:
    from use_cases import rejudge

    live = registered_names()
    seen, engines_seen, names_seen, prompts_seen, parsers_seen = {}, {}, {}, {}, {}
    choices_seen, budgets_seen, cuts_seen, grammars_seen, keys_seen = {}, {}, {}, {}, {}
    gone = set()
    remote_judge = False
    for name, logs in runs.items():
        ids, addresses, named, keys = set(), set(), set(), set()
        versions = {axis: set() for axis in rejudge.AXES}
        parsers = {axis: set() for axis in rejudge.AXES}
        choices = {axis: set() for axis in rejudge.AXES}
        budgets = {axis: set() for axis in rejudge.AXES}
        grammars = {axis: set() for axis in rejudge.AXES}
        # judged before the sampler was stamped: beside stamped rows the arm cannot say it held one sampler
        bare = set()
        cuts = 0
        for ql in logs:
            for axis in rejudge.AXES:
                stamp = ((ql.metrics or {}).get(axis) or {})
                if stamp.get("residency_id") is not None:
                    ids.add(stamp["residency_id"])
                remote_judge = remote_judge or stamp.get("residency_source") == NO_RESIDENCY
                read = engine_of(stamp, live)
                if read.address:
                    addresses.add(read.address)
                if read.name:
                    named.add(read.name)
                if read.state == DANGLING:
                    gone.add(read.name)
                # the version sits beside the model, in `prompts`, and never reached the stamp
                version = (ql.prompts or {}).get(f"judge_{axis}")
                if version is not None:
                    versions[axis].add(version)
                # a verdict stamped before the parser was recorded came from a local judge: no cut
                if stamp.get("model"):
                    parsers[axis].add(stamp.get("judge_parser") or answer_parsers.NO_PARSER)
                # a verdict from before the sampler was stamped says nothing, and silence is not a match
                sent = stamp.get("sampler")
                if isinstance(sent, dict):
                    choices[axis].add((sent.get("temperature"), sent.get("seed")))
                    budgets[axis].add(sent.get("max_tokens"))
                elif stamp.get("engine"):
                    bare.add(axis)
                cuts += bool(stamp.get(token_fields.JUDGE_CUT))
                if isinstance(stamp.get("engine_added"), dict):
                    grammars[axis].add(_grammar_of(stamp["engine_added"]))
                    keys.add(stamp["engine_added"].get("key_fingerprint"))
            # a cloud generator's key sits in the run snapshot, a cloud judge's in its verdict stamp
            for added in (((ql.metrics or {}).get("config") or {}).get("engine_added") or {}).values():
                if isinstance(added, dict):
                    keys.add(added.get("key_fingerprint"))
        keys_seen[name] = sorted(k for k in keys if k)
        seen[name] = sorted(ids)
        # the address, because it is the one field every era of this record carries
        engines_seen[name] = sorted(addresses)
        names_seen[name] = sorted(named)
        prompts_seen[name] = {axis: sorted(v) for axis, v in versions.items() if v}
        parsers_seen[name] = {axis: sorted(v) for axis, v in parsers.items() if v}
        choices_seen[name] = {axis: sorted(v, key=str) for axis, v in choices.items() if v and axis not in bare}
        budgets_seen[name] = {axis: sorted(v, key=str) for axis, v in budgets.items() if v and axis not in bare}
        cuts_seen[name] = cuts
        grammars_seen[name] = {axis: sorted(v) for axis, v in grammars.items() if v}
    # an arm that recorded nothing cannot agree with one that did: silence is not a match
    one = _all_agree(seen)
    one_engine, one_prompt = _all_agree(engines_seen), _one_ruler(prompts_seen)
    one_parser = _one_ruler(parsers_seen)
    one_sampler, one_budget = _one_ruler(choices_seen), _one_ruler(budgets_seen)
    one_grammar = _one_ruler(grammars_seen)
    one_name = _all_agree(names_seen)
    one_retrieval = _one_retrieval(runs)
    # a key is an account, not an instrument: two keys are one ruler billed twice, and the record says so
    spent_on = {key for held in keys_seen.values() for key in held}
    # silence is not a match: an arm with no recorded key leaves the question unread
    one_key = None if not spent_on or not all(keys_seen.values()) else len(spent_on) == 1
    return {
        "by_run": seen,
        "one_residency": one,
        "engines_by_run": engines_seen,
        "one_engine": one_engine,
        "engine_names_by_run": names_seen,
        "one_engine_name": one_name,
        # absent, not empty, when the table could not be read: silence is not "all were deleted"
        **({} if live is None else {"engines_gone": sorted(gone)}),
        "judge_prompts_by_run": prompts_seen,
        "one_judge_prompt": one_prompt,
        "judge_parsers_by_run": parsers_seen,
        "one_judge_parser": one_parser,
        "judge_grammars_by_run": grammars_seen,
        "one_judge_grammar": one_grammar,
        "judge_samplers_by_run": choices_seen,
        "one_judge_sampler": one_sampler,
        "judge_budgets_by_run": budgets_seen,
        "one_judge_budget": one_budget,
        "judge_cut_by_run": cuts_seen,
        # the sources and their ranks on the questions both arms answered, equal or not at all
        "one_deterministic": one_retrieval,
        "remote_judge": remote_judge,
        "broker_keys_by_run": keys_seen,
        "one_broker_key": one_key,
        "read_this_first": _what_to_read_first(
            one_engine, one_prompt, one, one_name, one_retrieval, one_parser=one_parser, one_grammar=one_grammar,
            remote_judge=remote_judge, one_sampler=one_sampler, one_budget=one_budget,
            cut=sum(cuts_seen.values()),
        ),
    }


def compare_runs(run_names: list[str]) -> dict:
    return compare({name: load_logs(name) for name in run_names})


if __name__ == "__main__":
    print(compare_runs(sys.argv[1:]))
