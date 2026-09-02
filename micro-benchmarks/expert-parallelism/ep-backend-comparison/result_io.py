"""Parse benchmark result markers despite interleaved native stdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


PREFIX = "ADAI_EP_RESULT "


def iter_result_objects(text: str, source: str = "<text>") -> Iterator[dict[str, Any]]:
    """Yield JSON objects following result markers in mixed process output.

    Native libraries can flush a diagnostic after Python has emitted a JSON
    object but before its newline reaches the combined log.  JSONDecoder's
    raw_decode identifies the exact end of the object without treating that
    trailing diagnostic as part of the result.
    """

    decoder = json.JSONDecoder()
    for line_number, line in enumerate(text.splitlines(), 1):
        search_from = 0
        while True:
            marker = line.find(PREFIX, search_from)
            if marker < 0:
                break
            payload_start = marker + len(PREFIX)
            try:
                result, consumed = decoder.raw_decode(line[payload_start:])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid benchmark result in {source}:{line_number}: {error}"
                ) from error
            if not isinstance(result, dict):
                raise ValueError(
                    f"benchmark result in {source}:{line_number} is not a JSON object"
                )
            yield result
            search_from = payload_start + consumed


def load_result_log(path: Path) -> list[dict[str, Any]]:
    return list(iter_result_objects(path.read_text(errors="replace"), str(path)))
