import engines
from models.registry import EngineKind, Placement

# the stand's engines as the seed registers them, for tests that need a spec and not a database
OLLAMA = engines.EngineSpec(1, "ollama", EngineKind.ollama, "OLLAMA", Placement.gpu)
VLLM = engines.EngineSpec(3, "vllm", EngineKind.vllm, "VLLM", Placement.gpu)
OLLAMA_CPU = engines.EngineSpec(5, "ollama-cpu", EngineKind.ollama, "OLLAMA_CPU", Placement.cpu)
VLLM_RERANK = engines.EngineSpec(7, "vllm-rerank", EngineKind.vllm, "VLLM_RERANK", Placement.gpu)
