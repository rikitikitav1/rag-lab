import re

# a fenced code block whole, its body in the group
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
# a line that opens or closes a fence
FENCE_LINE = re.compile(r"^\s*```")
# an inline code span, kept whole by a split so the prose around it can be rewritten alone
INLINE_CODE = re.compile(r"(`[^`\n]*`)")
# PDFium's mark for a hyphen it took for a word break; the route, the word rules and the code rows all read it
HYPHEN_MARK = "￾"
