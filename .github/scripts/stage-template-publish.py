#!/usr/bin/env python3
"""Validate and stage architecture assets for S3 template publishing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import tarfile
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


class CfnTagLoader(yaml.SafeLoader):
    """YAML loader that preserves structure while tolerating CloudFormation tags."""


def _construct_unknown(loader: CfnTagLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"Unsupported YAML node: {type(node).__name__}")


CfnTagLoader.add_multi_constructor("", lambda loader, _tag, node: _construct_unknown(loader, node))


S3_KEY_RE = re.compile(r"^[A-Za-z0-9!_.*'()/.-]+$")
PREFIX_REF_RE = re.compile(r"\$\{S3KeyPrefix\}([A-Za-z0-9._/-]+\.(?:ya?ml|sh))")
PUBLIC_URL_RE = re.compile(
    r"(?:https://awsome-distributed-ai\.s3\.amazonaws\.com/templates/|"
    r"s3://awsome-distributed-ai/templates/)"
    r"([A-Za-z0-9._/-]+\.(?:ya?ml|sh))"
)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=CfnTagLoader)


def is_cloudformation_template(path: Path) -> bool:
    try:
        document = load_yaml(path)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    return isinstance(document, dict) and isinstance(document.get("Resources"), dict)


def normalize_key(key: str) -> str:
    key = key.strip().lstrip("/")
    if not key:
        raise ValueError("empty S3 key")
    if ".." in Path(key).parts:
        raise ValueError(f"destination key must not contain '..': {key}")
    if not S3_KEY_RE.match(key):
        raise ValueError(f"destination key contains unsupported characters: {key}")
    return key


def load_manifest(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("manifest must be a YAML mapping")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a non-empty entries list")
    obsolete_keys = data.get("obsolete_keys", [])
    if not isinstance(obsolete_keys, list):
        raise ValueError("manifest obsolete_keys must be a list when present")
    return data


def architecture_namespace(source: Path) -> str | None:
    parts = source.parts
    try:
        architecture_index = parts.index("architectures")
    except ValueError:
        return None
    namespace_index = architecture_index + 1
    if namespace_index >= len(parts):
        return None
    return parts[namespace_index]


def validate_references(cfn_sources: list[Path], destination_keys: set[str]) -> list[str]:
    errors: list[str] = []
    for source in cfn_sources:
        text = source.read_text(encoding="utf-8")
        refs = set(PREFIX_REF_RE.findall(text)) | set(PUBLIC_URL_RE.findall(text))
        namespace = architecture_namespace(source)
        for ref in sorted(refs):
            namespaced_ref = f"{namespace}/{ref}" if namespace else None
            if ref not in destination_keys and namespaced_ref not in destination_keys:
                errors.append(f"{source}: references '{ref}', but no manifest entry publishes that key")
    return errors


def bundle_tar(source_dir: Path, patterns: list[str]) -> bytes:
    """An uncompressed tar of the files of source_dir matching patterns, identical for identical contents.

    Names sorted, mtime 0, uid/gid 0, mode 0644. Its sha256 names the bundle and is what the template
    pins and the bootstrap checks: compressed bytes depend on the zlib build, the tar does not.
    """
    files = sorted({f for pattern in patterns for f in source_dir.glob(pattern) if f.is_file()})
    links = [f for f in files if f.is_symlink()]
    if links:
        raise ValueError(f"{source_dir}: a bundle holds no symbolic links: {', '.join(str(f) for f in links)}")
    if not files:
        raise ValueError(f"{source_dir}: no files match {patterns}")
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            data = f.read_bytes()
            info = tarfile.TarInfo(f.relative_to(source_dir).as_posix())
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def bundle_bytes(tar: bytes) -> bytes:
    """The published tar.gz of a bundle_tar(), with no gzip name or time."""
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
        gz.write(tar)
    return out.getvalue()


def stage_bundle(entry: dict, repo_root: Path, stage_dir: Path) -> tuple[str, str]:
    """Stage one bundle entry under <key_prefix><sha256>.tar.gz; returns (key, sha256)."""
    source = (repo_root / entry["bundle"]).resolve()
    source.relative_to(repo_root)
    if not source.is_dir():
        raise ValueError(f"bundle source is not a directory: {entry['bundle']}")
    tar = bundle_tar(source, list(entry["files"]))
    digest = hashlib.sha256(tar).hexdigest()
    data = bundle_bytes(tar)
    key = normalize_key(f"{entry['key_prefix'].rstrip('/')}/{digest}.tar.gz")
    destination = stage_dir / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return key, digest


def write_list(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{file}\n" for file in files), encoding="utf-8")


def write_key_list(path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}\n" for key in keys), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--cfn-list", required=True, type=Path)
    parser.add_argument("--shell-list", required=True, type=Path)
    parser.add_argument("--obsolete-list", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--print-bundle-digests", action="store_true",
                        help="print '<bundle dir> <sha256>' for every bundle entry and exit")
    args = parser.parse_args()

    if args.print_bundle_digests:
        root = args.repo_root.resolve()
        for entry in load_manifest(args.manifest)["entries"]:
            if isinstance(entry, dict) and "bundle" in entry:
                tar = bundle_tar((root / entry["bundle"]).resolve(), list(entry["files"]))
                print(entry["bundle"], hashlib.sha256(tar).hexdigest())
        return 0

    repo_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    stage_dir = args.stage_dir.resolve()
    destination_to_source: dict[str, Path] = {}
    cfn_sources: list[Path] = []
    shell_sources: list[Path] = []
    obsolete_keys: list[str] = []
    errors: list[str] = []

    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(manifest["entries"], start=1):
        if not isinstance(entry, dict):
            errors.append(f"entry {index}: must be a mapping")
            continue
        if "bundle" in entry:
            # A content-addressed key never changes meaning, so an upload can never overwrite a
            # bundle a deployed stack pins.
            try:
                key, digest = stage_bundle(entry, repo_root, stage_dir)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"entry {index}: bundle: {exc}")
                continue
            if key in destination_to_source:
                errors.append(f"entry {index}: duplicate destination key '{key}'")
                continue
            destination_to_source[key] = repo_root / entry["bundle"]
            print(f"Bundle {entry['bundle']} -> {key} (sha256 {digest})")
            continue
        source_value = entry.get("source")
        key_value = entry.get("key")
        if not isinstance(source_value, str) or not isinstance(key_value, str):
            errors.append(f"entry {index}: source and key must be strings")
            continue

        try:
            key = normalize_key(key_value)
        except ValueError as exc:
            errors.append(f"entry {index}: {exc}")
            continue

        source = (repo_root / source_value).resolve()
        try:
            source.relative_to(repo_root)
        except ValueError:
            errors.append(f"entry {index}: source is outside repository: {source_value}")
            continue
        if not source_value.startswith("architectures/"):
            errors.append(f"entry {index}: source must be under architectures/: {source_value}")
            continue
        if not source.is_file():
            errors.append(f"entry {index}: source file does not exist: {source_value}")
            continue
        if key in destination_to_source:
            errors.append(
                f"entry {index}: duplicate destination key '{key}' also used by "
                f"{destination_to_source[key].relative_to(repo_root)}"
            )
            continue

        destination_to_source[key] = source

        if source.suffix in {".yaml", ".yml"} and is_cloudformation_template(source):
            cfn_sources.append(source)
        if source.suffix == ".sh":
            shell_sources.append(source)

        destination = stage_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for index, key_value in enumerate(manifest.get("obsolete_keys", []), start=1):
        if not isinstance(key_value, str):
            errors.append(f"obsolete key {index}: key must be a string")
            continue
        try:
            key = normalize_key(key_value)
        except ValueError as exc:
            errors.append(f"obsolete key {index}: {exc}")
            continue
        if key in destination_to_source:
            errors.append(f"obsolete key {index}: key is still published by entries: {key}")
            continue
        obsolete_keys.append(key)

    errors.extend(validate_references(cfn_sources, set(destination_to_source)))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    write_list(args.cfn_list, cfn_sources)
    write_list(args.shell_list, shell_sources)
    write_key_list(args.obsolete_list, obsolete_keys)
    print(f"Staged {len(destination_to_source)} files in {stage_dir}")
    print(f"Detected {len(cfn_sources)} CloudFormation templates")
    print(f"Detected {len(shell_sources)} shell scripts")
    print(f"Marked {len(obsolete_keys)} obsolete keys for deletion")
    print(f"Bucket: {manifest.get('bucket', 'awsome-distributed-ai')}")
    print(f"Prefix: {manifest.get('prefix', 'templates/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
