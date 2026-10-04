import re
from functools import lru_cache

import config
import logging_setup
from paths import ROOT

log = logging_setup.get_logger(__name__)

# a Latin token in a Russian question is a name the docs spell as is: a command, a flag, a product, `code`
_KEPT = re.compile(r"`[^`]+`|-{0,2}[A-Za-z][\w.:/+#-]*[\w+#]|-{0,2}[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def split_kept(question: str) -> tuple[str, list[str]]:
    kept = [m.group(0).strip("`") for m in _KEPT.finditer(question)]
    return re.sub(r"\s+", " ", _KEPT.sub(" ", question)).strip(), kept


# the keyword search reads words, not order, so the kept names ride beside the translation, never through it
def keyword_translation(question: str) -> str | None:
    cfg = config.settings.retrieval.keyword.translation
    if not cfg.enabled or not _CYRILLIC.search(question):
        return None
    words, kept = split_kept(question)
    if not _CYRILLIC.search(words):
        return " ".join(kept) or None
    try:
        translated = _translate(words, cfg.model_dir)
    except Exception as e:
        # a missing model leaves the search as it was rather than failing the question
        log.warning("translation.failed", error=str(e))
        return None
    return " ".join([translated, *kept]).strip() or None


def _translate(text: str, model_dir: str) -> str:
    translator, tokenizer, detokenizer = _model(model_dir)
    # a Marian model reads a sentence to its end marker; without it the decoder loops on its last words
    pieces = tokenizer.encode(text, out_type=str) + ["</s>"]
    result = translator.translate_batch([pieces], beam_size=2, max_decoding_length=128)
    return detokenizer.decode(result[0].hypotheses[0])


@lru_cache(maxsize=1)
def _model(model_dir: str):
    import ctranslate2
    import sentencepiece

    path = ROOT / model_dir
    translator = ctranslate2.Translator(str(path), device="cpu", compute_type="int8",
                                        inter_threads=1, intra_threads=2)
    return (translator, sentencepiece.SentencePieceProcessor(model_file=str(path / "source.spm")),
            sentencepiece.SentencePieceProcessor(model_file=str(path / "target.spm")))
