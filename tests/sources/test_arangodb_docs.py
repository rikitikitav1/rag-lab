from sources.arangodb_docs import ArangoDocsSource, without_shortcodes


# a tab's label, a key and an endpoint stay as words; a comment goes with its text; layout tags go alone
def test_shortcodes_leave_the_words_of_the_page():
    text = (
        '{{< tabs "view" >}}\n{{< tab "`search-alias` View" >}}\nUse an index.\n{{< /tab >}}\n{{< /tabs >}}\n'
        '{{< info >}}\nMind the limit of 64.\n{{< /info >}}\n'
        'Press {{< kbd "Tab" >}} now. {{< endpoint "DELETE" "/_api/collection/products" >}}\n'
        '{{% comment %}}\ninternal note\n{{% /comment %}}{{< image src="a.png" alt="icon">}}'
    )
    out = without_shortcodes(text)
    assert "`search-alias` View" in out and "Use an index." in out and "Mind the limit of 64." in out
    assert "Press Tab now." in out and "`DELETE /_api/collection/products`" in out
    assert "{{" not in out and "internal note" not in out and "a.png" not in out


# a Hugo page's title and lead live in its front matter; the lead opens the text, the fence never reaches it
def test_a_page_takes_its_title_from_the_front_matter(tmp_path):
    page = tmp_path / "syntax.md"
    page.write_text("---\ntitle: AQL Syntax\nweight: 5\ndescription: >-\n  Query types and\n  names\n---\n"
                    "## Query types\n\nAn AQL query returns.\n")
    parsed = ArangoDocsSource(tmp_path).read(page, "syntax.md")
    assert parsed.title == "AQL Syntax" and parsed.content.startswith("Query types and names\n\n## Query types")
