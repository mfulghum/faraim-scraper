from faraim_scraper.localize import extract_images, localize_images

BASE = "https://faraim.org"
PAGE = BASE + "/faa/aim/chapter-1/section-1-1-9.html"


def _html():
    from pathlib import Path

    return (Path(__file__).parent / "fixtures" / "aim_section.html").read_text(
        encoding="utf-8"
    )


def test_extract_images_same_host_only():
    urls = extract_images(_html(), PAGE, BASE)
    assert urls == [
        BASE + "/faa/aim/chapter-1/section-1-1-9-img-1.png",
        BASE + "/faa/aim/chapter-1/section-1-1-9-img-2.png",
    ]


def test_localize_images_rewrites_src_and_keeps_external():
    html, urls = localize_images(_html(), PAGE, BASE)
    assert 'src="section-1-1-9-img-1.png"' in html
    assert 'src="section-1-1-9-img-2.png"' in html
    assert 'src="https://cdn.example.com/tracker.gif"' in html
    assert len(urls) == 2


def test_localize_images_is_idempotent():
    once, urls1 = localize_images(_html(), PAGE, BASE)
    twice, urls2 = localize_images(once, PAGE, BASE)
    assert once == twice
    assert urls1 == urls2
