#!/usr/bin/env python3
"""Validate and stage architecture assets for S3 template publishing."""

from __future__ import annotations

import argparse
import re
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
    return data


def validate_references(cfn_sources: list[Path], destination_keys: set[str]) -> list[str]:
    errors: list[str] = []
    for source in cfn_sources:
        text = source.read_text(encoding="utf-8")
        refs = set(PREFIX_REF_RE.findall(text)) | set(PUBLIC_URL_RE.findall(text))
        for ref in sorted(refs):
            if ref not in destination_keys:
                errors.append(f"{source}: references '{ref}', but no manifest entry publishes that key")
    return errors


def write_list(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{file}\n" for file in files), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--cfn-list", required=True, type=Path)
    parser.add_argument("--shell-list", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    stage_dir = args.stage_dir.resolve()
    destination_to_source: dict[str, Path] = {}
    cfn_sources: list[Path] = []
    shell_sources: list[Path] = []
    errors: list[str] = []

    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(manifest["entries"], start=1):
        if not isinstance(entry, dict):
            errors.append(f"entry {index}: must be a mapping")
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

    errors.extend(validate_references(cfn_sources, set(destination_to_source)))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    write_list(args.cfn_list, cfn_sources)
    write_list(args.shell_list, shell_sources)
    print(f"Staged {len(destination_to_source)} files in {stage_dir}")
    print(f"Detected {len(cfn_sources)} CloudFormation templates")
    print(f"Detected {len(shell_sources)} shell scripts")
    print(f"Bucket: {manifest.get('bucket', 'awsome-distributed-ai')}")
    print(f"Prefix: {manifest.get('prefix', 'templates/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
