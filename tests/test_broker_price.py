import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "broker_price.py"


def _script():
    spec = importlib.util.spec_from_file_location("broker_price", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_lopsided_debits_give_the_page_prices_and_predict_the_third():
    # DeepSeek-V4-Flash on gonka, measured one call at a time against the balance
    rows = [
        {"model": "m", "case": "heavy_in", "in": 3022, "out": 2, "spent_usd": 3.03e-05},
        {"model": "m", "case": "heavy_out", "in": 20, "out": 1500, "spent_usd": 3.02e-05},
        {"model": "m", "case": "check", "in": 23, "out": 87, "spent_usd": 1.9e-06},
    ]
    got = _script().solve(rows)
    assert (round(got["in_per_million"], 2), round(got["out_per_million"], 2)) == (0.01, 0.02)
    assert abs(got["check_predicted_usd"] - got["check_spent_usd"]) <= 1e-7


def test_a_quoted_value_in_env_is_read_as_compose_reads_it():
    unquoted = _script()._unquoted
    assert (unquoted('"sk-1"'), unquoted("'sk-1'"), unquoted("sk-1"), unquoted('"')) == ("sk-1", "sk-1", "sk-1", '"')
