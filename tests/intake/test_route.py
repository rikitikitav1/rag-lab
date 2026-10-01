import pytest
from config import settings
from pypdf import PdfWriter
from use_cases import route

# the stand's own rules, named where a test reads with them rather than taken as a silent default
STAND = settings.intake.route


def _pages(*chars):
    return [{"page": n, "layer_chars": c} for n, c in enumerate(chars, start=1)]


def test_the_kind_is_read_from_the_file(tmp_path):
    assert route.route(tmp_path / "a.md", STAND) == [route.Run(None, None, "markdown")]
    assert route.route(tmp_path / "a.PNG", STAND) == [route.Run("mineru", None, "image")]
    assert route.route(tmp_path / "a.html", STAND) == [route.Run("docling", None, "html")]
    assert route.route(tmp_path / "a.docx", STAND)[0].engine == "docling"


def test_a_file_no_engine_reads_is_refused_by_its_suffix(tmp_path):
    for name, why in (("style.css", ".css"), ("content.OPF", ".opf"), ("font.ttf", ".ttf"), ("LICENSE", "no suffix")):
        with pytest.raises(route.Unsupported, match=why):
            route.route(tmp_path / name, STAND)


# a scanned book has no text layer, so every page goes to the OCR engine
def test_a_pdf_without_a_text_layer_goes_to_mineru(tmp_path):
    pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=595, height=842)
    writer.write(pdf)

    assert [(r.engine, r.pages) for r in route.route(pdf, STAND)] == [("mineru", (1, 4))]


# a scanned appendix is its own run; a blank page inside the text is not worth a handover of the card
def test_a_mixed_pdf_is_cut_into_runs_and_a_short_raster_run_stays_with_the_text():
    signals = _pages(900, 0, 800, 700, 0, 0, 0, 0, 600)

    runs = route._pdf_runs(signals, STAND)

    assert [(r.engine, r.pages) for r in runs] == [("docling", (1, 4)), ("mineru", (5, 8)), ("docling", (9, 9))]
    assert runs[1].why == "no text layer"
    assert runs[0].why == "text layer, 1 raster pages read by docling"
    assert runs[0].signals["absorbed"] == [2]


def test_code_crosses_a_break_only_when_both_edges_are_monospace():
    assert route.code_crosses([False, True], [True, False])
    assert not route.code_crosses([True, False], [True])
    assert not route.code_crosses([], [True])


# a piece's end steps off a break that code runs over, to the nearest clean break within the window
def test_a_piece_end_moves_off_a_break_that_code_runs_over(tmp_path, monkeypatch):
    import pypdfium2

    class _Page(int):
        def close(self):
            pass

    class _Document:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return _Page(i + 1)

        def close(self):
            pass

    # code runs over the break after each of these pages: the first piece ends at 11, the second finds no clean break
    crossed = {9, 10, 19, 20, 21, 22, 23}
    monkeypatch.setattr(pypdfium2, "PdfDocument", _Document)
    monkeypatch.setattr(route, "mono_rows", lambda page, margin: page)
    monkeypatch.setattr(route, "code_crosses", lambda before, after: before in crossed)

    import config

    moving = config.settings.intake.route.model_copy(update={"seam_window": 2})
    assert route.seamless_pieces(tmp_path / "a.pdf", (1, 30), 10, moving) == [(1, 11), (12, 21), (22, 30)]


# PDFium's line-end hyphen mark joins the word, as the converter prints it
def test_a_marked_line_end_hyphen_joins_the_word():
    assert route.joined_hyphens("независи\ufffe\r\nмо от") == "независимо от"
    assert route.joined_hyphens("sep\ufffearate") == "separate"
    # the pieces' layer joins its pages with a form feed, and a word broken at the page end spans it
    assert route.joined_hyphens("infor\ufffe\fmation") == "information"


def test_every_page_the_layer_reader_opens_is_closed_before_it_returns(tmp_path, monkeypatch):
    import pypdfium2

    open_at_close = []
    close = pypdfium2.PdfDocument.close

    def counting_close(document, *args, **kwargs):
        open_at_close.append(sum(1 for ref in document._kids if ref() is not None and ref().raw))
        return close(document, *args, **kwargs)

    monkeypatch.setattr(pypdfium2.PdfDocument, "close", counting_close)

    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(600, 800)
    with (tmp_path / "a.pdf").open("wb") as f:
        writer.write(f)
    route.layer_texts(tmp_path / "a.pdf")
    route.seamless_pieces(tmp_path / "a.pdf", (1, 4), 2, STAND)
    assert open_at_close and not any(open_at_close)
