from pypdf import PdfWriter
from use_cases import route


def _pages(*chars):
    return [{"page": n, "layer_chars": c} for n, c in enumerate(chars, start=1)]


def test_the_kind_is_read_from_the_file(tmp_path):
    assert route.route(tmp_path / "a.md") == [route.Run(None, None, "markdown")]
    assert route.route(tmp_path / "a.PNG") == [route.Run("mineru", None, "image")]
    assert route.route(tmp_path / "a.html") == [route.Run("docling", None, "html")]
    assert route.route(tmp_path / "a.docx")[0].engine == "docling"


# a scanned book has no text layer, so every page goes to the OCR engine
def test_a_pdf_without_a_text_layer_goes_to_mineru(tmp_path):
    pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=595, height=842)
    writer.write(pdf)

    assert [(r.engine, r.pages) for r in route.route(pdf)] == [("mineru", (1, 4))]


# a scanned appendix is its own run; a blank page inside the text is not worth a handover of the card
def test_a_mixed_pdf_is_cut_into_runs_and_a_short_raster_run_stays_with_the_text():
    signals = _pages(900, 0, 800, 700, 0, 0, 0, 0, 600)

    runs = route._pdf_runs(signals)

    assert [(r.engine, r.pages) for r in runs] == [("docling", (1, 4)), ("mineru", (5, 8)), ("docling", (9, 9))]
    assert runs[1].why == "no text layer"
    assert runs[0].why == "text layer, 1 raster pages read by docling"
    assert runs[0].signals["absorbed"] == [2]
