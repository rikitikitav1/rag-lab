"""Put the Cyrillic OCR model and CodeFormulaV2 into the image's model cache, refusing a wrong hash."""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# a Hugging Face revision is an immutable commit, so the weights alone are hashed
CODEFORMULA_FILES = [
    "added_tokens.json", "chat_template.json", "config.json", "generation_config.json", "model.safetensors",
    "preprocessor_config.json", "processor_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
]


def _get(url):
    with urllib.request.urlopen(url, timeout=1800) as response:
        return response.read()


def _check(data, sha256, what):
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.exit(f"{what}: sha256 {got}, expected {sha256}")


def main(models, easyocr_url, easyocr_sha256, codeformula_revision, codeformula_sha256):
    models = Path(models)
    weights = zipfile.ZipFile(io.BytesIO(_get(easyocr_url))).read("cyrillic_g2.pth")
    _check(weights, easyocr_sha256, "cyrillic_g2.pth")
    (models / "EasyOcr" / "cyrillic_g2.pth").write_bytes(weights)

    target = models / "docling-project--CodeFormulaV2"
    target.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/docling-project/CodeFormulaV2/resolve/{codeformula_revision}/"
    for name in CODEFORMULA_FILES:
        data = _get(base + name)
        if name == "model.safetensors":
            _check(data, codeformula_sha256, name)
        (target / name).write_bytes(data)


if __name__ == "__main__":
    main(*sys.argv[1:])
