from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _layer(tmp_path, text: str) -> str:
    path = tmp_path / "layer.yaml"
    path.write_text(text)
    return str(path)


def test_a_layer_replaces_the_whole_role_table():
    # merged, a judge without `engine` would inherit `vllm` from the main file and quietly break
    import config

    loaded = config._load(str(ROOT / "config.yaml"), str(ROOT / "config.cpu.yaml"))
    roles = loaded.llm.roles
    assert set(roles) == {"embedding", "generation", "judging", "paraphrasing", "ragas", "ragas_embedding"}
    assert {cfg.engine for cfg in roles.values()} == {"ollama-cpu"}
    assert "reranking" not in roles, "ollama scores no pairs, so the layer seats no reranker"


def test_a_layer_without_a_judge_or_with_more_than_roles_is_refused(tmp_path):
    import config

    base = str(ROOT / "config.yaml")
    no_judge = _layer(tmp_path, "llm:\n  roles:\n    generation: {model: m}\n    embedding: {model: e}\n")
    with pytest.raises(ValueError, match="drops"):
        config._load(base, no_judge)
    more = _layer(tmp_path, "llm:\n  roles: {}\n  context_length: 1\n")
    with pytest.raises(ValueError, match="nothing else"):
        config._load(base, more)


def test_the_seed_registers_the_processor_ollama():
    # on a clean database the no-card layout had no engine to seat its roles on
    import seed
    from models.registry import Placement

    added = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalar(self, _stmt):
            return False

        def add(self, row):
            added.append(row.name)

        def commit(self):
            pass

    original = seed.Session
    seed.Session = _Session
    try:
        seed.seed_engines()
    finally:
        seed.Session = original
    assert "ollama-cpu" in added
    import config

    assert {e.name: e.placement for e in config.settings.engines}["ollama-cpu"] == Placement.cpu


def test_a_role_on_an_ollama_the_pull_list_misses_gets_its_row(monkeypatch):
    # a role with `engine: ollama-cpu` was logged as absent and never seated
    import bootstrap
    import engines
    from models.registry import EngineKind, ModelRole, Placement

    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    added = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, _stmt):
            return SimpleNamespace(all=list)

        def scalar(self, _stmt):
            return None

        def add(self, row):
            added.append(row)

        def flush(self):
            for row in added:
                if getattr(row, "id", 0) is None:
                    row.id = 42

        def commit(self):
            pass

    monkeypatch.setattr(bootstrap, "Session", _Session)
    monkeypatch.setattr(bootstrap.engines, "spec_of_name", lambda name: cpu)
    monkeypatch.setattr(bootstrap.config.settings.llm, "roles",
                        {"judging": SimpleNamespace(model="qwen2.5:7b", engine="ollama-cpu")})
    monkeypatch.setattr(bootstrap.model_acceptance, "refuse_unfit_model", lambda *a: None)

    bootstrap._ensure_roles(None)

    model = next(row for row in added if not isinstance(row, ModelRole))
    assert (model.name, model.engine_id) == ("qwen2.5:7b", 5)
    assert [row.model_id for row in added if isinstance(row, ModelRole)] == [42]


def test_a_card_engine_down_points_to_the_no_card_mode_and_a_processor_one_does_not(monkeypatch):
    import engines
    from models.registry import EngineKind, Placement
    from use_cases import card_wait, stand_health

    vllm = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    monkeypatch.setattr(stand_health, "_roles", lambda: [
        ("judging", engines.Resolved("Qwen/Q", vllm)),
        ("generation", engines.Resolved("llama3.1:8b", cpu)),
    ])
    monkeypatch.setattr(stand_health, "engine_answers", lambda spec: False)
    monkeypatch.setattr(card_wait, "reranker_needed", lambda **kw: False)

    judging, generation = stand_health.roles_down()
    assert "scripts/up.sh --cpu" in judging
    assert "scripts/up.sh" not in generation


def test_an_empty_card_is_said_in_words(monkeypatch):
    from use_cases import stand_health

    monkeypatch.setattr(stand_health.card_holder, "on_card", lambda: [])
    monkeypatch.setattr(stand_health.engines, "card_engines", lambda kind=None: [])
    monkeypatch.setattr(stand_health.engines, "registered", lambda: [])

    assert stand_health.engines_section()["summary"] == "no engine holds the card"


def test_a_down_card_engine_on_a_stand_already_without_a_card_points_to_the_seat(monkeypatch):
    # the hint gave the very command the stand had been brought up with
    import engines
    from models.registry import EngineKind, Placement
    from use_cases import card_wait, stand_health

    vllm = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
    monkeypatch.setattr(stand_health, "_roles", lambda: [("judging", engines.Resolved("Qwen/Q", vllm))])
    monkeypatch.setattr(stand_health, "engine_answers", lambda spec: False)
    monkeypatch.setattr(card_wait, "reranker_needed", lambda **kw: False)
    monkeypatch.setattr(stand_health.config, "CONFIG_OVERLAY", "config.cpu.yaml")

    (judging,) = stand_health.roles_down()
    assert "PUT /v1/role" in judging and "--cpu" not in judging


def test_an_optional_role_nobody_seated_is_not_down_and_a_seated_one_whose_engine_fails_is(monkeypatch):
    # the paraphraser is used by two jobs; unseated it read as down, so readiness and the layer disagreed
    import config
    import engines
    from models.registry import EngineKind, Placement
    from use_cases import card_wait, stand_health

    assert "paraphrasing" not in config.REQUIRED_ROLES
    cpu = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
    roles = [("generation", engines.Resolved("llama3.1:8b", cpu)), ("paraphrasing", None),
             ("judging", None)]
    monkeypatch.setattr(stand_health, "_roles", lambda: roles)
    monkeypatch.setattr(stand_health, "engine_answers", lambda spec: True)
    monkeypatch.setattr(card_wait, "reranker_needed", lambda **kw: False)
    assert stand_health.roles_down() == ["judging: no model is seated"]

    roles[1] = ("paraphrasing", engines.Resolved("gemma2:9b", cpu))
    monkeypatch.setattr(stand_health, "engine_answers", lambda spec: False)
    assert "paraphrasing: ollama-cpu does not answer" in stand_health.roles_down()


class _Tagged(dict):
    pass


def _compose(name: str) -> dict:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def tagged(loader, node):
        value = loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else None
        return _Tagged(value or {})

    for tag in ("!reset", "!override"):
        Loader.add_constructor(tag, tagged)
    return yaml.load((ROOT / name).read_text(), Loader=Loader)


def test_a_env_filled_for_a_host_script_cannot_move_the_containers():
    # POSTGRES_HOST=localhost written as .env.example says would have sent three services to themselves
    main = _compose("docker-compose.yml")["services"]
    reading = {name for name, svc in main.items() if svc.get("env_file")}
    assert {"rag-lab", "bootstrap", "worker"} <= reading
    for name in reading:
        env = main[name].get("environment") or {}
        assert env.get("POSTGRES_HOST") == "" and env.get("OLLAMA_BASE_URL") == "", name


def test_every_service_that_mounts_the_tree_writes_no_bytecode():
    # the bootstrap ran as root on the mounted tree and left bytecode the host could not remove
    main = _compose("docker-compose.yml")["services"]
    mounting = {name for name, svc in main.items() if any(str(v).startswith(".:") for v in svc.get("volumes") or [])}
    assert {"bootstrap", "seed", "worker"} <= mounting
    for name in mounting:
        assert (main[name].get("environment") or {}).get("PYTHONDONTWRITEBYTECODE") == "1", name


def test_every_service_that_reserves_the_card_is_let_go_by_the_no_card_layer():
    # a new service with the card's reservation and no line in the layer would break `up.sh --cpu`
    main = _compose("docker-compose.yml")["services"]
    layer = _compose("docker-compose.cpu.yml")["services"]
    reserving = {name for name, svc in main.items() if "deploy" in svc and "profiles" not in svc}
    for name in reserving:
        over = layer.get(name, {})
        assert over.get("profiles") or isinstance(over.get("deploy"), _Tagged), name
    assert isinstance(layer["bootstrap"]["depends_on"], _Tagged), "the layer waits for no card engine"
    assert set(layer["bootstrap"]["depends_on"]) & {"ollama", "vllm"} == set()


def test_up_sh_looks_for_the_device_compose_reserves():
    import re

    devices = re.findall(r'device_ids: \["([^"]+)"\]', (ROOT / "docker-compose.yml").read_text())
    assert devices and set(devices) == {"nvidia.com/gpu=all"}
    assert f'grep -q "{devices[0]}"' in (ROOT / "scripts/up.sh").read_text()


def _run_up_sh(tmp_path, plan: str) -> list[str]:
    import os
    import subprocess

    log, stopped = tmp_path / "docker.log", tmp_path / "stopped"
    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text(f"""#!/bin/sh
echo "$*" >> {log}
case "$*" in
  info*) echo " nvidia.com/gpu=all" ;;
  "compose --dry-run up -d"*) [ "{plan}" = hang ] && sleep 30; printf '%s\\n' "{plan}" ;;
  "compose ps --status running --services") ;;
  "compose exec -T ollama ollama ps") printf 'NAME ID SIZE\\n'; [ -f {stopped} ] || echo "llama3.1:8b a1 6GB" ;;
  "compose exec -T ollama ollama stop"*) touch {stopped} ;;
esac
""")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}", "UP_PLAN_TIMEOUT": "1"}
    subprocess.run(["bash", str(ROOT / "scripts/up.sh")], env=env, check=True, capture_output=True, timeout=20)
    return log.read_text().splitlines()


def test_up_sh_frees_the_card_before_a_vllm_start_and_leaves_ollama_alone_otherwise(tmp_path):
    # recreated while ollama held a model on the card, vLLM died short of memory and the stack waited
    calls = _run_up_sh(tmp_path, "Container rag-lab-vllm-1  Recreate")
    stop = calls.index("compose exec -T ollama ollama stop llama3.1:8b")
    assert calls[-1] == "compose up -d" and stop < len(calls) - 1

    quiet = tmp_path / "quiet"
    quiet.mkdir()
    calls = _run_up_sh(quiet, "Container rag-lab-vllm-1  Running")
    assert not [c for c in calls if "ollama" in c], "a running vLLM takes nothing from ollama"
    assert calls[-1] == "compose up -d"



def test_up_sh_does_not_hang_on_a_plan_that_never_ends_and_still_frees_the_card(tmp_path):
    # after `down` the dry run waited on a one-shot container's exit for ever, silently
    calls = _run_up_sh(tmp_path, "hang")
    assert "compose exec -T ollama ollama stop llama3.1:8b" in calls, "a vLLM that is not running is about to start"
    assert calls[-1] == "compose up -d"


def test_up_sh_cpu_stops_the_card_engines_a_running_stand_kept_answering_from(tmp_path):
    import os
    import subprocess

    log = tmp_path / "docker.log"
    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho "$*" >> {log}\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"}
    subprocess.run(["bash", str(ROOT / "scripts/up.sh"), "--cpu"], env=env, check=True, capture_output=True, timeout=20)
    calls = log.read_text().splitlines()
    assert calls[0] == "compose stop ollama vllm" and calls[-1].endswith("up -d")

def _judge_model() -> str:
    import config

    return config._load(str(ROOT / "config.yaml")).llm.roles["judging"].model


def test_up_sh_starts_vllm_with_the_judge_config_yaml_names(tmp_path):
    # the judge was named twice, in config.yaml and in compose, and only the boot noticed they differed
    import os
    import subprocess

    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    seen = tmp_path / "seen"
    fake.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  info*) echo ' nvidia.com/gpu=all' ;;\n"
        f"  'compose up -d'*) echo \"$VLLM_MODEL\" > {seen} ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "VLLM_MODEL"}
    env["PATH"] = f"{fake.parent}:{env['PATH']}"
    subprocess.run(["bash", str(ROOT / "scripts/up.sh")], env=env, check=True, capture_output=True)
    assert seen.read_text().strip() == _judge_model()


def test_the_compose_default_mirrors_the_judge_config_yaml_names():
    import re

    text = (ROOT / "docker-compose.yml").read_text()
    (default,) = re.findall(r'"--model", "\$\{VLLM_MODEL:-([^}]+)\}"', text)
    assert default == _judge_model(), "a bare `docker compose up` would start another judge"



def test_the_window_is_one_number_in_every_place_that_states_it():
    # six places said 8192 and only a comment tied them: one changed alone truncates without a word
    import re

    compose = (ROOT / "docker-compose.yml").read_text()
    stated = {int(v) for v in re.findall(r"\$\{(?:OLLAMA_CONTEXT_LENGTH|VLLM_MAX_MODEL_LEN):-(\d+)\}", compose)}
    stated |= {int(v) for v in re.findall(r'"--max-model-len", "(\d+)"', compose)}
    stated |= {int(v) for v in re.findall(
        r"^(?:OLLAMA_CONTEXT_LENGTH|VLLM_MAX_MODEL_LEN)=(\d+)$",
        (ROOT / ".env.example").read_text(), re.M,
    )}
    stated.add(yaml.safe_load((ROOT / "config.yaml").read_text())["llm"]["context_length"])
    assert len(stated) == 1, f"the window is stated as {sorted(stated)}"

def test_the_judge_is_an_optional_dependency_of_the_boot_and_ollama_is_not():
    # compose waits for an optional healthy dependency while it starts and goes on once it has died
    main = _compose("docker-compose.yml")["services"]["bootstrap"]["depends_on"]
    assert main["vllm"] == {"condition": "service_healthy", "required": False}
    assert main["ollama"].get("required", True) is True, "the generator and the embedder live there"
