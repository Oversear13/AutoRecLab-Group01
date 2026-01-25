import re

_OPEN_FENCE = re.compile(r"^\s*```[\w+-]*\s*\n")
_CLOSE_FENCE = re.compile(r"\n\s*```\s*$")


def strip_markdown_fences(code: str) -> str:
    # Remove ```lang\n ... \n``` or ```\n ... \n```
    code = _OPEN_FENCE.sub("", code)
    code = _CLOSE_FENCE.sub("", code)
    return code.strip()


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string
