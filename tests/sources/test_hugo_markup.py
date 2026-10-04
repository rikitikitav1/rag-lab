from use_cases.markdown_cleanup import as_indexed, without_hugo_shortcodes


# a tab's label, a key and an endpoint stay as words; a comment goes with its text; layout tags go alone
def test_shortcodes_leave_the_words_of_the_page():
    text = (
        '{{< tabs "view" >}}\n{{< tab "`search-alias` View" >}}\nUse an index.\n{{< /tab >}}\n{{< /tabs >}}\n'
        '{{< info >}}\nMind the limit of 64.\n{{< /info >}}\n'
        'Press {{< kbd "Tab" >}} now. {{< endpoint "DELETE" "/_api/collection/products" >}}\n'
        '{{% comment %}}\ninternal note\n{{% /comment %}}{{< image src="a.png" alt="icon">}}'
    )
    out = without_hugo_shortcodes(text)
    assert "`search-alias` View" in out and "Use an index." in out and "Mind the limit of 64." in out
    assert "Press Tab now." in out and "`DELETE /_api/collection/products`" in out
    assert "{{" not in out and "internal note" not in out and "a.png" not in out


# what Kubernetes and Redis pages show of their shortcodes: a term, a section heading, a feature state, a fence
def test_the_shortcodes_of_other_hugo_sites_render_what_the_site_shows():
    text = (
        'A {{< glossary_tooltip text="Pod" term_id="pod" >}} runs.\n'
        '{{% heading "whatsnext" %}}\n'
        '{{< feature-state for_k8s_version="v1.30" state="beta" >}}\n'
        '{{< highlight bash >}}\nredis-cli PING\n{{< /highlight >}}\n'
        'See [the guide]({{< relref "/develop/guide" >}}).\n'
    )
    out = without_hugo_shortcodes(text)
    assert "A Pod runs." in out and "## What's next" in out and "FEATURE STATE: Kubernetes v1.30 [beta]" in out
    assert "```bash\nredis-cli PING\n```" in out and "[the guide]()" in out and "{{" not in out


# the markup is rendered only where the source declares it
def test_the_markup_is_read_only_where_it_is_declared():
    page = "---\ntitle: AQL Syntax\n---\nPress {{< kbd \"Tab\" >}}.\n"
    assert "{{<" in as_indexed(page) and "Press Tab." in as_indexed(page, "hugo")


# an MDN reference shows its label or its name, a badge its word, and the page furniture goes
def test_mdn_macros_render_what_the_page_shows():
    from use_cases.markdown_cleanup import without_mdn_macros

    text = ('Call {{domxref("Element.click()")}} or {{domxref("Element/click", "click()")}}'
            ' on {{HTMLElement("input")}}. {{optional_inline}} See {{glossary("CORS")}}'
            ' and {{HTTPStatus("404")}} per {{RFC(7231)}}.\n'
            '{{Compat}}\n{{EmbedLiveSample("Examples", 300, 200)}}')
    assert without_mdn_macros(text) == ("Call `Element.click()` or `click()` on `<input>`. (optional) See CORS and 404"
                                        " per RFC 7231.\n\n")
