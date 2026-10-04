# the counter each markdown rule writes under its knob: one place holds both names
KNOB_COUNTERS = {
    "decode_entities": "entities_decoded",
    "drop_lone_pipes": "pipes_dropped",
    "outline_levels": "relevelled",
    "numbered_levels": "numbered_levels",
    "picture_addresses": "pictures_addressed",
    "formula_text": "formulas_from_layer",
    "man_page_titles": "man_page_titles",
    "unescape_bullets": "bullets_unescaped",
    "unescape_underscores": "underscores_unescaped",
    "demote_caption_headings": "captions_demoted",
    "drop_running_headings": "running_heads_dropped",
    "restore_dashes": "dashes_restored",
    "join_broken_words": "words_joined",
    "join_split_words": "split_words_joined",
    "drop_inherited_members": "inherited_members_dropped",
    "drop_repeated_code": "repeated_code_dropped",
}
# steps every reading runs, with no knob of their own
ALWAYS = (
    "rebuilt", "kept", "joined", "tables_joined", "fenced", "paragraphs_joined", "rows_run_on", "rows_once",
    "duplicates_dropped",
)
# knobs that set how a step works or which route a file takes, not a rule that edits the markdown by itself
NOT_RULES = frozenset(
    {"splice_tables", "headings_by_number", "listing_callouts", "mono_by_step", "html_one_title", "join_layer_hyphens",
     "epub_chapters", "contents_outline"}
)
COUNTERS = (*ALWAYS, *KNOB_COUNTERS.values())


# a rule runs only under its knob and must say how often it fired, so a silent rule is an error and not a zero
class Fired:
    def __init__(self, rule):
        self.rule = rule
        self.counts: dict[str, int] = {}

    def run(self, knob: str, markdown: str, step, *args) -> str:
        if not getattr(self.rule, knob):
            return markdown
        out = step(markdown, *args)
        if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], int)):
            raise TypeError(f"the {knob} rule gave back no count")
        counter = KNOB_COUNTERS[knob]
        self.counts[counter] = self.counts.get(counter, 0) + out[1]
        return out[0]
