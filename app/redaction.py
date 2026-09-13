import os
import re

# the secrets the stand holds only in its environment; a shorter value would cut ordinary words
_NAMES = re.compile(r"^(.+_(API_KEY|KEY|TOKEN|SECRET|PASSWORD)|HF_TOKEN)$")
_MIN = 8
_IN_QUERY = re.compile(r"([?&](?:key|api_key|apikey|token|access_token)=)[^&\s\"']+", re.I)


# the value is known exactly, so it is cut by value: a broker quoting it back leaves no trace
def redact(text: str) -> str:
    if not text:
        return text
    for name, value in os.environ.items():
        if value and len(value) >= _MIN and is_secret(name):
            text = text.replace(value, "<key>")
    return _IN_QUERY.sub(r"\1<key>", text)


# a name the integrations are told is secret is one here too, whatever it is called
def is_secret(name: str) -> bool:
    import config

    return bool(_NAMES.match(name)) or name in config.settings.mcp_integrations.secret_env
