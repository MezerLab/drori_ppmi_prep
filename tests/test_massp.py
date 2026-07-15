import json
from urllib.parse import parse_qs, urlparse

from drori_ppmi_prep.segmentation import massp


class _FakeResponse:
    def __init__(self, files):
        self._files = files

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self._files).encode("utf-8")


def test_figshare_file_download_url_searches_paginated_results(monkeypatch):
    page_size = 100
    first_page = [
        {"name": f"other_{index}.nii.gz", "download_url": f"https://example.org/{index}"}
        for index in range(page_size)
    ]
    target = {
        "name": "ahead-massp2_avg-bestlabel_decade-61to80.nii.gz",
        "download_url": "https://example.org/atlas.nii.gz",
    }
    calls = []

    def fake_urlopen(url):
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        page = int(query["page"][0])
        if page == 1:
            return _FakeResponse(first_page)
        return _FakeResponse([target])

    monkeypatch.setattr(massp, "urlopen", fake_urlopen)

    download_url = massp._figshare_file_download_url("27292209", target["name"])

    assert download_url == target["download_url"]
    assert len(calls) == 2
