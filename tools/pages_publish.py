"""Publish one authenticated documentation subtree without executing build artifacts.

The trusted workflow runs this module from its default-branch commit. Artifacts
are bounded ZIP data; Git's bare object/index plumbing never checks them out.
Source-ref checks and the gh-pages fast-forward are separate operations: a ref
moving in that final network window fails the post-push check, not an atomic
cross-ref transaction. A per-ref generation ledger prevents older publications.
"""

import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

Json = dict[str, Any]

REPOSITORY = "opthub-org/pytorch-bsf"
WORKFLOW_PATH = ".github/workflows/sphinx-pages-build.yml"
WORKFLOW_NAME = "GHPages build"
CHECK_NAME = "GHPages publish"
LEDGER = ".pages-publish"
MAX_ARCHIVE_BYTES = 100_000_000
MAX_SITE_BYTES = 250_000_000
MAX_FILE_BYTES = 20_000_000
MAX_FILES = 20_000
MAX_MANIFEST_BYTES = 5_000_000
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
TAG = re.compile(r"v\d+\.\d+\.\d+")
RESERVED = {".git", ".gitattributes", ".gitmodules", LEDGER, ".nojekyll", "readme.md", "index.html", "cname"}


class GitHubError(RuntimeError):
    """An API failure without a response body or credential in the message."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"GitHub API returned HTTP {status}")


class StripRedirectCredentials(urllib.request.HTTPRedirectHandler):
    """Never forward the API authorization header to artifact storage."""

    def redirect_request(
        self, request: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Artifact redirect must use HTTPS")
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    """Small repository-scoped API client; all pagination is explicit."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.opener = urllib.request.build_opener(StripRedirectCredentials())

    def request(self, path: str, method: str = "GET", data: Any = None, *, binary: bool = False) -> Any:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{REPOSITORY}" + (f"/{path}" if path else ""),
            data=None if data is None else json.dumps(data).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"},
            method=method,
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                body = response.read((MAX_ARCHIVE_BYTES if binary else 20_000_000) + 1)
        except urllib.error.HTTPError as error:
            raise GitHubError(error.code) from None
        limit = MAX_ARCHIVE_BYTES if binary else 20_000_000
        if len(body) > limit:
            raise ValueError("GitHub response exceeds its size limit")
        return body if binary else json.loads(body)

    def pages(self, path: str, key: str | None = None) -> list[Json]:
        result = []
        for page in range(1, 101):
            separator = "&" if "?" in path else "?"
            response = self.request(f"{path}{separator}per_page=100&page={page}")
            rows = response if key is None else response[key]
            if not isinstance(rows, list):
                raise ValueError("Malformed paginated API response")
            result.extend(rows)
            if len(rows) < 100:
                return result
        raise ValueError("API pagination exceeds its limit")


def positive_int(value: Any) -> int:
    """Reject booleans and coercions in service and artifact identifiers."""
    if type(value) is not int or value <= 0:
        raise ValueError("Expected a positive integer identifier")
    return value


def safe_path(value: Any) -> str:
    """Validate a relative POSIX data path without normalizing away evidence."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid relative artifact path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.casefold() in {".git", ".gitattributes", ".gitmodules", LEDGER} for part in parts):
        raise ValueError("Unsafe artifact path component")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Control character in artifact path")
    return value


def destination(ref: Any) -> str:
    """Retain existing version URLs for allowed branches and release tags."""
    if not isinstance(ref, str) or not ref.startswith(("refs/heads/", "refs/tags/")):
        raise ValueError("Unsupported source ref")
    name = ref.split("/", 2)[2]
    safe_path(name)
    if any(part.startswith(".") or part.endswith(".lock") for part in name.split("/")):
        raise ValueError("Invalid Git ref component")
    if name.endswith(".") or ".." in name or "@{" in name or any(c in name for c in " ~^:?*[\\"):
        raise ValueError("Invalid Git ref name")
    if name.split("/", 1)[0].casefold() in RESERVED:
        raise ValueError("Reserved publication namespace")
    if ref.startswith("refs/tags/"):
        if TAG.fullmatch(name) is None:
            raise ValueError("Only release tags are published")
    elif name.startswith(("gh-pages", "release-please")):
        raise ValueError("Excluded documentation branch")
    return name


def overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def trusted_workflow() -> bytes:
    """Read policy from the trusted checkout, never the artifact or producer."""
    return (Path(__file__).resolve().parents[1] / ".github/pages-build-template.yml").read_bytes()


def trusted_legacy_owners() -> dict[str, str]:
    """Read the reviewed ownership census from the trusted control checkout."""
    path = Path(__file__).resolve().parents[1] / ".github/pages-legacy-owners.json"
    raw = path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Legacy ownership policy is too large")
    policy = json.loads(raw, object_pairs_hook=unique_json)
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", "gh_pages_sha", "owners", "reserved_owners"}
        or type(policy["schema"]) is not int
        or policy["schema"] != 1
        or not isinstance(policy["gh_pages_sha"], str)
        or SHA.fullmatch(policy["gh_pages_sha"]) is None
        or not isinstance(policy["owners"], dict)
        or not 0 < len(policy["owners"]) <= MAX_FILES
        or not isinstance(policy["reserved_owners"], dict)
        or len(policy["reserved_owners"]) > MAX_FILES
    ):
        raise ValueError("Invalid legacy ownership policy")
    if set(policy["owners"]) & set(policy["reserved_owners"]):
        raise ValueError("Observed and reserved ownership roots must be distinct")
    owners: dict[str, str] = {**policy["owners"], **policy["reserved_owners"]}
    for root, ref in owners.items():
        if destination(ref) != root:
            raise ValueError("Legacy ownership root and ref disagree")
    roots = sorted(owners)
    if any(overlap(left, right) for index, left in enumerate(roots) for right in roots[index + 1 :]):
        raise ValueError("Legacy publication namespaces overlap")
    return owners


def check_namespace(ref: str, paths: list[str], ledgers: list[Json], owners: dict[str, str]) -> None:
    """Reject unknown existing prefixes and every other legacy version owner."""
    target = destination(ref)
    for root, owner in owners.items():
        if overlap(root, target) and owner != ref:
            raise ValueError("Trusted legacy publication URLs overlap")
    owned = owners.get(target) == ref or any(record["source_ref"] == ref for record in ledgers)
    if not owned and any(path == target or path.startswith(target + "/") for path in paths):
        raise ValueError("Existing publication prefix has no trusted owner")
    # Catch roots introduced after the census, including archived versions that
    # no longer have a live Git ref or a publisher generation ledger.
    roots = {path.rsplit("/", 1)[0] for path in paths if path.endswith(("/objects.inv", "/.buildinfo"))}
    if any(overlap(root, target) and (root != target or not owned) for root in roots):
        raise ValueError("Unregistered legacy publication URLs overlap")


def authenticate_run(api: Any, event: Json) -> tuple[Json, Json]:
    """Bind the webhook to canonical service metadata before any check write."""
    run = event["workflow_run"]
    ident, attempt = positive_int(run["id"]), positive_int(run["run_attempt"])
    canonical = api.request(f"actions/runs/{ident}/attempts/{attempt}")
    latest = api.request(f"actions/runs/{ident}")
    repo = api.request("")
    workflow = api.request("actions/workflows/" + WORKFLOW_PATH.rsplit("/", 1)[1])
    if repo["full_name"] != REPOSITORY or repo["default_branch"] != "master":
        raise ValueError("Unexpected repository policy")
    for record in (run, canonical, latest):
        if (
            record["id"] != ident
            or record["run_attempt"] != attempt
            or record["repository"]["id"] != repo["id"]
            or record["head_repository"]["id"] != repo["id"]
            or record["repository"]["full_name"] != REPOSITORY
            or record["head_repository"]["full_name"] != REPOSITORY
            or record["workflow_id"] != workflow["id"]
            or record["path"] != WORKFLOW_PATH
            or record["name"] != WORKFLOW_NAME
            or record["event"] not in {"push", "workflow_dispatch"}
            or record["status"] != "completed"
            or record["conclusion"] != "success"
            or record["head_sha"] != canonical["head_sha"]
            or record["head_branch"] != canonical["head_branch"]
            or record["run_number"] != canonical["run_number"]
        ):
            raise ValueError("Untrusted, stale, or unsuccessful producer run")
    if workflow["path"] != WORKFLOW_PATH or SHA.fullmatch(canonical["head_sha"]) is None:
        raise ValueError("Invalid producer workflow or commit")
    definition = api.request(f"contents/{WORKFLOW_PATH}?ref={canonical['head_sha']}")
    if (
        definition["type"] != "file"
        or definition["encoding"] != "base64"
        or type(definition["size"]) is not int
        or not 0 < definition["size"] <= 100_000
        or base64.b64decode("".join(definition["content"].split()), validate=True) != trusted_workflow()
    ):
        raise ValueError("Producer workflow differs from trusted read-only build policy")
    positive_int(canonical["run_number"])
    return canonical, repo


def load_artifact(api: Any, run: Json, repo: Json) -> tuple[Json, dict[str, bytes], Json]:
    """Download one immutable attempt-named artifact and validate its ZIP data."""
    name = f"pages-{run['id']}-{run['run_attempt']}"
    artifacts = [a for a in api.pages(f"actions/runs/{run['id']}/artifacts", "artifacts") if a["name"] == name]
    if len(artifacts) != 1:
        raise ValueError("Expected exactly one producer artifact")
    artifact = api.request(f"actions/artifacts/{positive_int(artifacts[0]['id'])}")
    owner = artifact["workflow_run"]
    if (
        artifact["id"] != artifacts[0]["id"]
        or artifact["name"] != name
        or artifact["expired"] is not False
        or type(artifact["size_in_bytes"]) is not int
        or not 0 < artifact["size_in_bytes"] <= MAX_ARCHIVE_BYTES
        or not isinstance(artifact["digest"], str)
        or DIGEST.fullmatch(artifact["digest"]) is None
        or owner["id"] != run["id"]
        or owner["repository_id"] != repo["id"]
        or owner["head_repository_id"] != repo["id"]
        or owner["head_sha"] != run["head_sha"]
        or owner["head_branch"] != run["head_branch"]
        or artifact["created_at"] < run["run_started_at"]
    ):
        raise ValueError("Artifact metadata does not match its producer attempt")
    raw = api.request(f"actions/artifacts/{artifact['id']}/zip", binary=True)
    if len(raw) != artifact["size_in_bytes"] or "sha256:" + hashlib.sha256(raw).hexdigest() != artifact["digest"]:
        raise ValueError("Artifact archive size or digest mismatch")
    manifest, files = validate_archive(raw)
    expected = {"schema": 1, "repository": REPOSITORY, "run_id": run["id"], "run_attempt": run["run_attempt"]}
    if any(type(manifest.get(key)) is not type(value) or manifest[key] != value for key, value in expected.items()):
        raise ValueError("Manifest producer identity mismatch")
    if manifest["source_sha"] != run["head_sha"] or destination(manifest["source_ref"]) != run["head_branch"]:
        raise ValueError("Manifest source does not match the producer run")
    return manifest, files, artifact


def unique_json(pairs: list[tuple[str, Any]]) -> Json:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def archive_entry(entry: zipfile.ZipInfo, archive: zipfile.ZipFile) -> tuple[str, bytes | None]:
    """Validate one ZIP entry before reading bounded bytes."""
    if entry.filename != entry.orig_filename:
        raise ValueError("Archive path was truncated")
    name = safe_path(entry.filename[:-1] if entry.is_dir() else entry.filename)
    kind = stat.S_IFMT(entry.external_attr >> 16)
    if kind not in ({0, stat.S_IFDIR} if entry.is_dir() else {0, stat.S_IFREG}):
        raise ValueError("Archive contains a link or special file")
    if name != "manifest.json" and name != "site" and not name.startswith("site/"):
        raise ValueError("Unexpected archive namespace")
    if entry.is_dir():
        if entry.file_size:
            raise ValueError("Directory entry contains data")
        return name, None
    limit = MAX_MANIFEST_BYTES if name == "manifest.json" else MAX_FILE_BYTES
    if entry.file_size > limit:
        raise ValueError("Expanded artifact file is too large")
    if entry.flag_bits & 1 or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ValueError("Unsupported archive encoding")
    with archive.open(entry) as stream:
        content = stream.read(limit + 1)
    if len(content) != entry.file_size or len(content) > limit:
        raise ValueError("Invalid archive entry size")
    return name, content


def read_archive(raw: bytes) -> dict[str, bytes]:
    """Bound archive-wide metadata and preserve original entry names."""
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive is too large")
    contents = {}
    seen = set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_FILES * 2:
            raise ValueError("Too many archive entries")
        if sum(entry.file_size for entry in entries) > MAX_SITE_BYTES:
            raise ValueError("Expanded artifact is too large")
        for entry in entries:
            name, content = archive_entry(entry, archive)
            if name in seen:
                raise ValueError("Duplicate archive path")
            seen.add(name)
            if content is not None:
                contents[name] = content
    return contents


def validate_archive(raw: bytes) -> tuple[Json, dict[str, bytes]]:
    """Read regular files only and validate a closed, hashed manifest."""
    contents = read_archive(raw)
    manifest_bytes = contents.pop("manifest.json", None)
    files = {safe_path(name[5:]): content for name, content in contents.items() if name.startswith("site/")}
    if (
        manifest_bytes is None
        or not files
        or len(files) != len(contents)
        or len(files) > MAX_FILES
        or "index.html" not in files
    ):
        raise ValueError("Artifact needs a manifest and a nonempty HTML site")
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_json)
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "repository",
        "run_id",
        "run_attempt",
        "source_ref",
        "source_sha",
        "files",
    }:
        raise ValueError("Invalid manifest schema")
    records = manifest["files"]
    if not isinstance(records, dict) or set(records) != set(files):
        raise ValueError("Manifest file inventory mismatch")
    for name, content in files.items():
        record = records[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int
            or record["size"] != len(content)
            or record["sha256"] != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError("Manifest file size or hash mismatch")
    if any("/".join(name.split("/")[:count]) in files for name in files for count in range(1, len(name.split("/")))):
        raise ValueError("File and directory path collision")
    return manifest, files


def peel_commit(api: Any, value: Json) -> str:
    for _ in range(8):
        if value["type"] == "commit":
            return str(value["sha"])
        if value["type"] != "tag":
            break
        value = api.request(f"git/tags/{value['sha']}")["object"]
    raise ValueError("Source ref is not a commit or bounded annotated tag")


def live_refs(api: Any, manifest: Json) -> None:
    """Authenticate the full ref kind and check all live URL namespaces."""
    ref, sha = manifest["source_ref"], manifest["source_sha"]
    name = destination(ref)
    if not isinstance(sha, str) or SHA.fullmatch(sha) is None:
        raise ValueError("Invalid source SHA")
    refs = api.request("git/matching-refs/heads/") + api.request("git/matching-refs/tags/")
    selected = [row for row in refs if row["ref"] == ref]
    if len(selected) != 1:
        raise ValueError("Source ref was deleted or is ambiguous")
    if peel_commit(api, selected[0]["object"]) != sha:
        raise ValueError("Source ref moved since the build")
    for row in refs:
        if row["ref"] == ref:
            continue
        try:
            other = destination(row["ref"])
        except ValueError:
            continue
        if overlap(name, other):
            raise ValueError("Live version URLs overlap")


class GitPages:
    """Edit only a bare repository index; artifact bytes never become code."""

    def __init__(self, root: Path, token: str) -> None:
        self.root = root
        self.token = token
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_ATTR_NOSYSTEM="1")
        self.env.update(GIT_AUTHOR_NAME="GitHub Action", GIT_AUTHOR_EMAIL="action@github.com")
        self.env.update(GIT_COMMITTER_NAME="GitHub Action", GIT_COMMITTER_EMAIL="action@github.com")
        self.git("init", "--bare", "--quiet")
        self.git("fetch", "--quiet", "--depth=1", "--no-tags", f"https://github.com/{REPOSITORY}.git", "refs/heads/gh-pages")
        self.base = self.git("rev-parse", "FETCH_HEAD").decode().strip()
        self.git("read-tree", self.base)

    def git(self, *args: str, data: bytes | None = None, authenticated: bool = False) -> bytes:
        env = dict(self.env)
        if authenticated:
            credential = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
            env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader")
            env["GIT_CONFIG_VALUE_0"] = "AUTHORIZATION: basic " + credential
        process = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.attributesFile=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "--git-dir",
                str(self.root),
                *args,
            ],
            input=data,
            env=env,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if process.returncode:
            raise RuntimeError("Bare Git publication failed; the branch may have changed")
        return process.stdout

    def paths(self) -> list[str]:
        return self.git("ls-tree", "-r", "--name-only", "-z", self.base).decode().split("\0")[:-1]

    def ledgers(self) -> list[Json]:
        result = []
        for path in self.paths():
            if path.startswith(LEDGER + "/"):
                content = self.git("show", f"{self.base}:{path}")
                if len(content) > MAX_MANIFEST_BYTES:
                    raise ValueError("Oversized publication ledger")
                record = json.loads(content, object_pairs_hook=unique_json)
                if path != ledger_path(record["source_ref"]):
                    raise ValueError("Invalid publication ledger path")
                destination(record["source_ref"])
                generation(record)
                if (
                    not isinstance(record.get("files"), list)
                    or len(record["files"]) > MAX_FILES
                    or len(set(record["files"])) != len(record["files"])
                ):
                    raise ValueError("Invalid managed file inventory")
                for name in record["files"]:
                    safe_path(name)
                result.append(record)
        return result

    def prepare(self, manifest: Json, files: dict[str, bytes], record: Json) -> str:
        target = destination(manifest["source_ref"])
        ledgers = self.ledgers()
        paths = self.paths()
        check_namespace(manifest["source_ref"], paths, ledgers, trusted_legacy_owners())
        # Keep unknown old files within an authenticated owner's version, and
        # delete only files previously managed by this same publisher/ref.
        removed = []
        for old in ledgers:
            if old["source_ref"] == manifest["source_ref"]:
                removed = [target + "/" + name for name in old["files"]]
        existing = set(paths) - set(removed)
        for name in files:
            path = target + "/" + name
            if any("/".join(path.split("/")[:count]) in existing for count in range(1, len(path.split("/")))):
                raise ValueError("Existing publication file blocks the destination")
            if any(other.startswith(path + "/") for other in existing):
                raise ValueError("New file would replace retained publication directories")
        if removed:
            deleted = b"".join(b"0 " + b"0" * 40 + b"\t" + path.encode() + b"\0" for path in removed)
            self.git("update-index", "-z", "--index-info", data=deleted)
        updates = {target + "/" + name: content for name, content in files.items()}
        updates[ledger_path(manifest["source_ref"])] = json.dumps(record, sort_keys=True).encode() + b"\n"
        for path, content in updates.items():
            blob = self.git("hash-object", "-w", "--no-filters", "--stdin", data=content).strip()
            self.git("update-index", "--add", "-z", "--index-info", data=b"100644 " + blob + b"\t" + path.encode() + b"\0")
        tree = self.git("write-tree").decode().strip()
        message = f"Publish {manifest['source_ref']} {manifest['source_sha']}\n".encode()
        return self.git("commit-tree", tree, "-p", self.base, data=message).decode().strip()

    def push(self, commit: str) -> None:
        # The commit is a child of base. The lease additionally rejects a remote
        # force-push to an ancestor, which an ordinary fast-forward would allow.
        self.git(
            "push",
            "--quiet",
            f"--force-with-lease=refs/heads/gh-pages:{self.base}",
            f"https://github.com/{REPOSITORY}.git",
            f"{commit}:refs/heads/gh-pages",
            authenticated=True,
        )


def ledger_path(ref: str) -> str:
    return f"{LEDGER}/{hashlib.sha256(ref.encode()).hexdigest()}.json"


def generation(record: Json) -> tuple[int, ...]:
    return tuple(positive_int(record[key]) for key in ("run_number", "run_id", "run_attempt"))


def publish(api: Any, event: Json, pages_factory: Callable[[], Any]) -> Json:
    """Authenticate, validate, check freshness, then publish a single version."""
    run, repo = authenticate_run(api, event)
    check = api.request(
        "check-runs",
        "POST",
        {
            "name": CHECK_NAME,
            "head_sha": run["head_sha"],
            "status": "in_progress",
            "external_id": f"pages-{run['id']}-{run['run_attempt']}",
            "details_url": os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            + f"/{REPOSITORY}/actions/runs/{os.environ.get('GITHUB_RUN_ID', run['id'])}",
        },
    )
    try:
        manifest, files, artifact = load_artifact(api, run, repo)
        live_refs(api, manifest)
        record = {key: manifest[key] for key in ("source_ref", "source_sha", "run_id", "run_attempt")}
        record.update(run_number=run["run_number"], artifact_id=artifact["id"], digest=artifact["digest"], files=sorted(files))
        pages = pages_factory()
        for old in pages.ledgers():
            if old["source_ref"] == manifest["source_ref"]:
                if generation(old) > generation(record):
                    raise ValueError("A newer generation is already published")
                if generation(old) == generation(record) and old != record:
                    raise ValueError("Publication generation was reused with different data")
            elif overlap(destination(old["source_ref"]), destination(manifest["source_ref"])):
                raise ValueError("Retained publication URLs overlap")
        commit = pages.prepare(manifest, files, record)
        # Re-fetch service metadata, not just a candidate manifest or an old webhook.
        authenticate_run(api, event)
        live_refs(api, manifest)
        pages.push(commit)
        # No API transaction can compare the source ref and gh-pages together.
        live_refs(api, manifest)
    except Exception:
        api.request(
            f"check-runs/{check['id']}",
            "PATCH",
            {
                "status": "completed",
                "conclusion": "failure",
                "output": {
                    "title": "Documentation publication failed",
                    "summary": "Publication failed authentication, validation or freshness checks. Inspect the publisher run.",
                },
            },
        )
        raise
    api.request(
        f"check-runs/{check['id']}",
        "PATCH",
        {
            "status": "completed",
            "conclusion": "success",
            "output": {
                "title": "Documentation published",
                "summary": (
                    f"Published `{manifest['source_ref']}` at `{manifest['source_sha']}` from artifact {artifact['id']}."
                ),
            },
        },
    )
    return record


def main() -> None:
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY or os.environ.get("GITHUB_EVENT_NAME") != "workflow_run":
        raise ValueError("Publisher requires its trusted workflow_run context")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    api = GitHub(os.environ["GH_TOKEN"])
    with tempfile.TemporaryDirectory(prefix="pages-bare-") as directory:
        publish(api, event, lambda: GitPages(Path(directory), api.token))


if __name__ == "__main__":
    main()
