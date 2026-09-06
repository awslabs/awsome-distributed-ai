"""Fail closed on the installed transfer package, independently of the image tag."""
import importlib.metadata as metadata
import json
import pathlib
import sglang


def inspect_image():
    versions = {name: metadata.version(name) for name in ("sglang", "nixl", "nixl-cu13")}
    assert versions["nixl"] == versions["nixl-cu13"] == "1.1.0", versions
    assert versions["sglang"] == "0.5.12.post1", versions
    root = pathlib.Path(sglang.__file__).parent
    env = (root / "srt/environ.py").read_text()
    args = (root / "srt/server_args.py").read_text()
    conn = (root / "srt/disaggregation/nixl/conn.py").read_text()
    assert 'SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")' in env
    assert 'disaggregation_transfer_backend: str = "mooncake"' in args
    assert "self.agent.create_backend(backend, backend_params)" in conn
    return {"installed_versions": versions, "source_defaults": {"sglang": "mooncake", "nixl": "UCX"},
            "explicit_make_connection_call_in_connector": "make_connection(" in conn or "makeConnection(" in conn}

if __name__ == "__main__":
    print(json.dumps(inspect_image(), indent=2))
