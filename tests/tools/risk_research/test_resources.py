import io
import json
from email.message import Message

import pytest

from tools.risk_research import fetch_resources as resources


def _catalogue(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resources": [
                    {
                        "id": "sample",
                        "title": "Sample",
                        "filename": "sample.pdf",
                        "kind": "pdf",
                        "urls": ["https://example.org/sample.pdf"],
                    }
                ],
            }
        )
    )
    return path


class _Response(io.BytesIO):
    def __init__(self, body, content_type="application/pdf", url="https://example.org/sample.pdf"):
        super().__init__(body)
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self):
        return self.url


def test_download_rejects_html_and_keeps_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "sample.pdf"
    target.write_bytes(b"%PDF-existing")
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: _Response(b"<html>blocked</html>"))
    with pytest.raises(ValueError, match="PDF header"):
        resources._download("https://example.org/sample.pdf", target, 1)
    assert target.read_bytes() == b"%PDF-existing"
    assert list(tmp_path.iterdir()) == [target]


def test_download_rejects_wrong_content_type_and_size(tmp_path, monkeypatch):
    target = tmp_path / "sample.pdf"
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: _Response(b"<html>", "text/html"))
    with pytest.raises(ValueError, match="content type"):
        resources._download("https://example.org/sample.pdf", target, 1)
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: _Response(b"%PDF-" + b"a" * 10))
    with pytest.raises(ValueError, match="size limit"):
        resources._download("https://example.org/sample.pdf", target, 1, maximum_bytes=10)
    assert not list(tmp_path.iterdir())


def test_public_http_requires_explicit_opt_in(tmp_path, monkeypatch):
    target = tmp_path / "sample.pdf"
    calls = []

    def open_response(*args, **kwargs):
        calls.append(args[0].full_url)
        return _Response(b"%PDF-public-notes", url="http://example.org/sample.pdf")

    monkeypatch.setattr(resources.urllib.request, "urlopen", open_response)
    with pytest.raises(ValueError, match="HTTPS"):
        resources._download("http://example.org/sample.pdf", target, 1)
    assert not calls
    with pytest.raises(ValueError, match="redirected"):
        resources._download("https://example.org/sample.pdf", target, 1)
    assert not target.exists()
    assert resources._download("http://example.org/sample.pdf", target, 1, allow_http=True) == "application/pdf"
    assert target.read_bytes() == b"%PDF-public-notes"


def test_catalogue_http_opt_in_reaches_the_downloader(tmp_path, monkeypatch):
    catalogue = _catalogue(tmp_path)
    data = json.loads(catalogue.read_text())
    data["resources"][0]["urls"] = ["http://example.org/sample.pdf"]
    catalogue.write_text(json.dumps(data))
    destination = tmp_path / "downloads"
    with pytest.raises(ValueError, match="HTTPS"):
        resources.fetch_resources(catalogue, destination)
    assert not destination.exists()
    data["resources"][0]["allow_http"] = True
    catalogue.write_text(json.dumps(data))

    def download(url, path, timeout, *, allow_http):
        assert allow_http is True
        path.write_bytes(b"%PDF-public-notes")
        return "application/pdf"

    monkeypatch.setattr(resources, "_download", download)
    monkeypatch.setattr(resources, "inspect_pdf", lambda _: {"pages": 1, "text_status": "searchable"})
    result = resources.fetch_resources(catalogue, destination)
    assert result["resources"][0]["status"] == "ready"


def test_checksum_resume_and_corruption_repair(tmp_path, monkeypatch):
    catalogue = _catalogue(tmp_path)
    destination = tmp_path / "downloads"
    calls = []

    def download(url, path, timeout):
        calls.append(url)
        path.write_bytes(b"%PDF-public-test")
        return "application/pdf"

    monkeypatch.setattr(resources, "_download", download)
    monkeypatch.setattr(resources, "inspect_pdf", lambda _: {"pages": 1, "text_status": "searchable"})
    first = resources.fetch_resources(catalogue, destination)
    second = resources.fetch_resources(catalogue, destination)
    assert len(calls) == 1
    assert second["resources"][0]["sha256"] == first["resources"][0]["sha256"]
    (destination / "sample.pdf").write_bytes(b"corrupted")
    repaired = resources.fetch_resources(catalogue, destination)
    assert len(calls) == 2
    assert repaired["resources"][0]["sha256"] == first["resources"][0]["sha256"]


def test_failed_download_remains_retryable(tmp_path, monkeypatch):
    catalogue = _catalogue(tmp_path)
    destination = tmp_path / "downloads"

    def fail(*_):
        raise OSError("temporary outage")

    monkeypatch.setattr(resources, "_download", fail)
    result = resources.fetch_resources(catalogue, destination)
    assert result["resources"][0]["status"] == "error"
    assert "temporary outage" in result["resources"][0]["errors"][0]["error"]


def test_cached_download_uses_current_catalogue_metadata(tmp_path, monkeypatch):
    catalogue = _catalogue(tmp_path)
    destination = tmp_path / "downloads"
    calls = []

    def download(url, path, timeout):
        calls.append(url)
        path.write_bytes(b"%PDF-public-test")
        return "application/pdf"

    monkeypatch.setattr(resources, "_download", download)
    monkeypatch.setattr(resources, "inspect_pdf", lambda _: {"pages": 1, "text_status": "searchable"})
    first = resources.fetch_resources(catalogue, destination)
    data = json.loads(catalogue.read_text())
    data["resources"][0].update(title="Corrected title", authors="Corrected authors", scope="Survey only")
    catalogue.write_text(json.dumps(data))
    changed = resources.fetch_resources(catalogue, destination)["resources"][0]
    assert changed["title"] == "Corrected title"
    assert changed["authors"] == "Corrected authors"
    assert changed["scope"] == "Survey only"
    assert changed["sha256"] == first["resources"][0]["sha256"]
    assert len(calls) == 1


def test_path_traversal_is_rejected_before_creating_files(tmp_path):
    catalogue = _catalogue(tmp_path)
    data = json.loads(catalogue.read_text())
    data["resources"][0]["filename"] = "../outside.pdf"
    catalogue.write_text(json.dumps(data))
    destination = tmp_path / "downloads"
    with pytest.raises(ValueError, match="directory"):
        resources.fetch_resources(catalogue, destination)
    assert not destination.exists()
