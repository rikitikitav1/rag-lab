import os
import re

# the secrets the stand holds only in its environment; a shorter value would cut ordinary words
_NAMES = re.compile(r"^(.+_API_KEY|HF_TOKEN)$")
_MIN = 8
_IN_QUERY = re.compile(r"([?&](?:key|api_key|apikey|token|access_token)=)[^&\s\"']+", re.I)


# the value is known exactly, so it is cut by value: a broker quoting it back leaves no trace
def redact(text: str) -> str:
    if not text:
        return text
    for name, value in os.environ.items():
        if value and len(value) >= _MIN and _NAMES.match(name):
            text = text.replace(value, "<key>")
    return _IN_QUERY.sub(r"\1<key>", text)
