"""Fail closed on the installed transfer package, independently of the image tag."""
import importlib.metadata as metadata
import json
import pathlib
import ast
import sglang


def inspect_image():
    versions = {name: metadata.version(name) for name in ("sglang", "nixl", "nixl-cu13")}
    pairs = {
        ("0.5.12.post1", "1.1.0", "1.1.0"): "mainline",
        ("0.5.19", "1.4.1", "1.4.1"): "v4-optional",
    }
    pair = tuple(versions[name] for name in ("sglang", "nixl", "nixl-cu13"))
    assert pair in pairs, versions
    profile = pairs[pair]
    root = pathlib.Path(sglang.__file__).parent
    env = (root / "srt/environ.py").read_text()
    args = (root / "srt/server_args.py").read_text()
    conn = (root / "srt/disaggregation/nixl/conn.py").read_text()
    assert 'SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")' in env
    # The optional engine uses Annotated fields; inspect the assigned value.
    defaults = [node.value.value for node in ast.walk(ast.parse(args))
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == 'disaggregation_transfer_backend'
                and isinstance(node.value, ast.Constant)]
    assert defaults == ['mooncake'], defaults
    assert "self.agent.create_backend(backend, backend_params)" in conn
    return {"engine_profile": profile, "installed_versions": versions, "source_defaults": {"sglang": "mooncake", "nixl": "UCX"},
            "explicit_make_connection_call_in_connector": "make_connection(" in conn or "makeConnection(" in conn}

if __name__ == "__main__":
    print(json.dumps(inspect_image(), indent=2))
