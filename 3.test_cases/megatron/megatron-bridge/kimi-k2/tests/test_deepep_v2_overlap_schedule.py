#!/usr/bin/env python3
"""Regression gate for DeepEP v2 fine-grained pipeline scheduling semantics."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from megatron.core.models.common.utils import should_free_input
from megatron.core.models.gpt.fine_grained_callables import (
    build_transformer_layer_callables,
)


def config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        cuda_graph_modules=[],
        fp4=None,
        fp8=None,
        moe_flex_dispatcher_backend=backend,
        moe_token_dispatcher_type="flex",
    )


def main() -> None:
    for backend in ("deepep", "deepep_v2"):
        assert not should_free_input("moe_dispatch", True, config(backend), 12)

    source = inspect.getsource(build_transformer_layer_callables)
    assert 'in ("deepep", "deepep_v2")' in source
    print(
        "DEEPEP_V2_OVERLAP_SCHEDULE_PASS "
        + json.dumps(
            {
                "backends_with_deepep_state_restore": ["deepep", "deepep_v2"],
                "moe_dispatch_input_retained": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
