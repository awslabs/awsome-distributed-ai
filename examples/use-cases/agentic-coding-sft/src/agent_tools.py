# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tool schemas + system prompt shared by training data prep and the agent eval.

No heavy imports: also loaded inside the vLLM image.
"""

# SWE-agent's tool set, written as JSON schemas. The dataset stores calls but not the schemas.
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command in the repository environment and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": ("View, create, and edit files. `view` shows a file or directory, "
                        "`create` writes a new file, `str_replace` replaces one exact occurrence "
                        "of old_str with new_str, `insert` adds new_str after insert_line, "
                        "`undo_edit` reverts the last edit."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
            "path": {"type": "string", "description": "Absolute path to a file or directory."},
            "file_text": {"type": "string", "description": "Content for `create`."},
            "old_str": {"type": "string", "description": "Exact text to replace (`str_replace`)."},
            "new_str": {"type": "string", "description": "Replacement or inserted text."},
            "insert_line": {"type": "integer", "description": "Line number for `insert`."},
            "view_range": {"type": "array", "items": {"type": "integer"},
                           "description": "Optional [start, end] lines for `view`."}},
            "required": ["command", "path"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Submit the changes once the task is complete.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = ("You are a coding agent working in a Python repository. Use the tools to explore the "
          "code, make the required change, verify it, and call `submit` when done.")