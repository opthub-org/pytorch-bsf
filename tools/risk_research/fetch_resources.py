"""Fetch public research reading into a local, untracked resource directory.

Run ``python -m tools.risk_research.fetch_resources`` from the checkout.
The catalogue deliberately distinguishes full books, preprints and samples.
Downloads stay local: inclusion in the catalogue grants no redistribution rights.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CATALOGUE = Path(__file__).with_name("resources.json")
DEFAULT_DESTINATION = Path(__file__).resolve().parents[2] / ".research/skeleton-risk/resources"
MAX_BYTES = 100 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def _validate_resource(resource: dict[str, Any]) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", resource["id"]):
        raise ValueError("resource id must be a simple slug")
    filename = resource["filename"]
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("resource filename must not contain a directory")
    if resource["kind"] != "pdf" or not filename.endswith(".pdf"):
        raise ValueError("only PDF resources are supported")
    allow_http = resource.get("allow_http", False)
    if type(allow_http) is not bool:
        raise ValueError("allow_http must be a boolean")
    schemes = ("https://", "http://") if allow_http else ("https://",)
    if not resource["urls"] or any(not url.startswith(schemes) for url in resource["urls"]):
        raise ValueError("resource URLs must use HTTPS unless allow_http is explicitly enabled")


def _download(url: str, destination: Path, timeout: float, maximum_bytes: int = MAX_BYTES, *, allow_http: bool = False) -> str:
    schemes = ("https://", "http://") if allow_http else ("https://",)
    if not url.startswith(schemes):
        raise ValueError("download URL must use HTTPS unless allow_http is explicitly enabled")
    request = urllib.request.Request(url, headers={"User-Agent": "pytorch-bsf-risk-research/1.0"})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not response.geturl().startswith(schemes):
                raise ValueError("download redirected away from permitted HTTP schemes")
            content_type = str(response.headers.get_content_type())
            if content_type not in {"application/pdf", "application/octet-stream"}:
                raise ValueError(f"unexpected content type: {content_type}")
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) > maximum_bytes:
                raise ValueError("download exceeds size limit")
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                for block in iter(lambda: response.read(65536), b""):
                    if size == 0 and not block.startswith(b"%PDF-"):
                        raise ValueError("download does not have a PDF header")
                    size += len(block)
                    if size > maximum_bytes:
                        raise ValueError("download exceeds size limit")
                    stream.write(block)
                if not size:
                    raise ValueError("empty download")
        temporary.replace(destination)
        return content_type
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inspect_pdf(path: Path) -> dict[str, Any]:
    """Record PDF metadata and extract searchable text when Poppler is available."""
    result: dict[str, Any] = {}
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        info = subprocess.run([pdfinfo, str(path)], check=True, capture_output=True, text=True, timeout=60).stdout
        for line in info.splitlines():
            key, _, value = line.partition(":")
            if key == "Pages":
                result["pages"] = int(value.strip())
            elif key == "Title" and value.strip():
                result["pdf_title"] = value.strip()
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        result["text_status"] = "pdftotext unavailable"
        return result
    text_path = path.with_suffix(".txt")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        subprocess.run([pdftotext, "-layout", str(path), str(temporary)], check=True, capture_output=True, timeout=60)
        text = temporary.read_text(encoding="utf-8")
        result["text_characters"] = len(text)
        result["text_status"] = "searchable" if text.strip() else "empty"
        result["text_file"] = text_path.name
        result["text_sha256"] = _sha256(temporary)
        temporary.replace(text_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _fetch_one(resource: dict[str, Any], destination: Path, old: dict[str, Any], timeout: float) -> dict[str, Any]:
    path = destination / resource["filename"]
    record = {**resource, "status": "error"}
    reusable = old.get("filename") == resource["filename"] and old.get("urls") == resource["urls"]
    if reusable and old.get("status") == "ready" and path.exists() and _sha256(path) == old.get("sha256"):
        acquisition_fields = (
            "downloaded_from",
            "downloaded_at",
            "sha256",
            "bytes",
            "content_type",
            "retrieval_method",
            "retrieval_evidence",
        )
        record.update({key: old[key] for key in acquisition_fields if key in old})
        record["status"] = "ready"
    else:
        errors = []
        for url in resource["urls"]:
            try:
                if resource.get("allow_http", False):
                    content_type = _download(url, path, timeout, allow_http=True)
                else:
                    content_type = _download(url, path, timeout)
                record.update(
                    status="ready",
                    downloaded_from=url,
                    downloaded_at=datetime.now(UTC).isoformat(),
                    sha256=_sha256(path),
                    bytes=path.stat().st_size,
                    content_type=content_type,
                )
                break
            except (OSError, ValueError, urllib.error.URLError) as error:
                errors.append({"url": url, "error": str(error)})
        if record["status"] != "ready":
            record["errors"] = errors
    if record["status"] == "ready":
        try:
            record.update(inspect_pdf(path))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record["inspection_error"] = str(error)
    return record


def fetch_resources(catalogue: Path, destination: Path, timeout: float = 60) -> dict[str, Any]:
    """Fetch missing or modified PDFs and write checksummed acquisition evidence."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    data = json.loads(catalogue.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported catalogue schema")
    resources = data["resources"]
    for resource in resources:
        _validate_resource(resource)
    ids = [resource["id"] for resource in resources]
    filenames = [resource["filename"] for resource in resources]
    if len(set(ids)) != len(ids) or len(set(filenames)) != len(filenames):
        raise ValueError("resource ids and filenames must be unique")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    previous_by_id = {record["id"]: record for record in previous.get("resources", [])}
    records: list[dict[str, Any]] = []
    manifest = {
        "schema_version": 1,
        "catalogue_sha256": _sha256(catalogue),
        "updated_at": datetime.now(UTC).isoformat(),
        "resources": records,
        "additional_access": data.get("additional_access", []),
    }
    for resource in resources:
        records.append(_fetch_one(resource, destination, previous_by_id.get(resource["id"], {}), timeout))
        _atomic_json(manifest_path, manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    """Fetch the public catalogue and print a concise acquisition summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    try:
        manifest = fetch_resources(args.catalogue, args.destination, args.timeout)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for record in manifest["resources"]:
        print(f"{record['id']}: {record['status']}; text={record.get('text_status', 'unavailable')}")
    print(f"Manifest: {args.destination / 'manifest.json'}")
    return int(any(record["status"] != "ready" for record in manifest["resources"]))


if __name__ == "__main__":
    raise SystemExit(main())
