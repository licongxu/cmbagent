# gpt-oss-120b: Structured Output & Tool Calling Differences vs OpenAI API

Notes on running `gpt-oss-120b` via vLLM with CMBAgent/AG2, compared to
OpenAI hosted models (GPT-4.1, etc.).

## Server Configuration

```bash
python server.py   # GPU 1, port 8011
```

Key vLLM flags:

| Flag | Value | Purpose |
|------|-------|---------|
| `--tool-call-parser` | `openai` | Parses Harmony protocol tool calls |
| `--reasoning-parser` | `openai_gptoss` | Separates analysis/final channels |
| `--structured-outputs-config.backend` | `auto` | Guided decoding for Pydantic schemas |
| `--structured-outputs-config.reasoning_parser` | `openai_gptoss` | Schema constraints only on final channel |
| `--generation-config` | `auto` | Uses model's EOS tokens incl. `<\|call\|>` |

---

## 1. Structured Output

### OpenAI API (GPT-4.1)

- Send `response_format=PydanticModel`.
- API guarantees valid JSON matching the schema.
- Response goes directly into `message.content` as a JSON string.
- OpenAI client auto-parses via `message.parsed`.

### gpt-oss-120b (vLLM)

- Same API interface (`response_format` with `json_schema`).
- vLLM uses **guided decoding** (xgrammar/outlines) to constrain generation.
- The model produces **reasoning** in `<|channel|>analysis` before the
  structured output in `<|channel|>final`.
- `--reasoning-parser openai_gptoss` separates them:
  - `message.reasoning_content` = analysis channel (chain-of-thought)
  - `message.content` = final channel (the JSON)
- `--structured-outputs-config.reasoning_parser` tells guided decoding to
  only enforce schema constraints on the final channel.

### Differences that matter

| Aspect | OpenAI API | gpt-oss-120b (vLLM) |
|--------|-----------|---------------------|
| Reasoning | Not exposed | `message.reasoning_content` populated |
| JSON quality | Always valid | May contain un-escaped LaTeX backslashes (`\frac`, `\sum`) |
| Token count | Lower (no reasoning overhead) | Higher (reasoning + final) |
| Latency | Network-bound | Lower (local GPU), but reasoning adds tokens |

### Impact on AG2 / CMBAgent

- AG2's `client.py` calls `model_validate_json(content)` -- works the same.
- `save_final_plan` in CMBAgent receives a JSON **string** (not a Pydantic
  object or markdown). The default `_parse_plan_string` expects markdown
  and produces empty `sub_tasks`. **Requires patching** to try
  `json.loads()` first, with a fallback for LaTeX backslashes
  (`re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)`) and `strict=False` for
  control characters.

---

## 2. Tool Calling

### OpenAI API (GPT-4.1)

- Send `tools=[...]` with function definitions.
- Model returns `message.tool_calls = [{id, function: {name, arguments}}]`.
- `arguments` is always valid JSON.
- `message.content` is `None` when tool calls are present.
- `message.tool_calls` is `None` (not `[]`) when no calls are made.

### gpt-oss-120b (vLLM)

- Uses the **Harmony protocol** internally:
  ```
  <|start|>assistant to=functions.my_func<|channel|>commentary<|message|>{"arg":"val"}<|call|>
  ```
- The `openai` tool parser translates Harmony tokens into standard
  OpenAI `tool_calls` format.
- `<|call|>` (token 200012) is an EOS token (`--generation-config auto`
  loads this from `generation_config.json`).

### Differences that matter

| Aspect | OpenAI API | gpt-oss-120b (vLLM) |
|--------|-----------|---------------------|
| No tool call | `tool_calls=None` | `tool_calls=[]` (empty list) |
| Reasoning-only response | N/A | `content=None`, `reasoning_content` populated |
| Tool arg JSON | Always valid | May contain LaTeX that breaks `json.loads` |
| Forced tool choice | Content may be present | Content often empty (reasoning goes to analysis channel) |

### Impact on AG2

- **`tool_calls=[]` vs `None`**: AG2 checks `tool_calls is not None` (line
  390 in `client.py`). Empty list `[]` passes this check, so AG2 treats it
  as a tool-call response. But there are no actual calls, and `content` is
  also empty, causing:
  ```
  ValueError: Message can't be converted into a valid ChatCompletion message.
  Either content or function_call must be provided.
  ```
  **Fix**: Strip empty `tool_calls=[]` and fall back to `reasoning_content`.

- **JSON parse error in tool arguments**: vLLM's `openai_tool_parser.py`
  logs `Error decoding JSON tool call from response` when arguments contain
  LaTeX. This is **non-fatal** -- vLLM catches the exception and returns
  the raw text. The request still gets a 200 OK.

---

## 3. Other Differences

### Temperature

AG2 sends `temperature=1e-05`. vLLM clamps it to `0.01` with a warning:
```
temperature 1e-05 is less than 0.01, which may cause numerical errors
```
This is cosmetic and does not affect functionality.

### Token naming

gpt-oss uses special tokens not found in standard OpenAI models:

| Token | ID | Purpose |
|-------|------|---------|
| `<\|start\|>` | 200006 | Message start |
| `<\|end\|>` | 200007 | Message end |
| `<\|channel\|>` | 200005 | Channel selector (analysis/final) |
| `<\|message\|>` | 200008 | Content start |
| `<\|call\|>` | 200012 | Tool call terminator (also EOS) |
| `<\|return\|>` | 200002 | Tool return / EOS |
| `<\|endoftext\|>` | 199999 | Standard EOS |

### Wrong tool call parser

`--tool-call-parser seed_oss` does **NOT** work for gpt-oss. `seed_oss`
expects XML-style `<seed:tool_call>` tokens from the Seed model family.
Use `--tool-call-parser openai` which handles the Harmony protocol.

---

## 4. Required Notebook Patches for CMBAgent

Two monkey-patches are needed when using gpt-oss-120b with CMBAgent
(applied in the notebook, not in the cmbagent source):

### Patch 1: `save_final_plan` (JSON plan parsing)

#### The problem

In the `deep_research` workflow, the planning phase produces a structured
plan via the `planner_response_formatter` agent. This agent sets
`response_format = PlannerResponse` (a Pydantic model). After AG2 gets the
LLM response, it calls `PlannerResponse.model_validate_json(content)` and
then `.format()`, which converts the Pydantic object into a markdown
string. This formatted string is stored in `final_context["final_plan"]`.

When `save_final_plan()` is called, it checks the type of
`final_context["final_plan"]`:
- **Pydantic object** → calls `.model_dump()` → correct dict
- **dict/list** → uses directly → correct
- **string** → calls `_parse_plan_string()` which expects markdown like:
  ```
  - Step 1:
      * sub-task: ...
      * agent in charge: researcher
  ```

With **GPT-4.1**, the `.format()` method produces clean markdown, and
`_parse_plan_string` parses it correctly.

With **gpt-oss-120b**, the `.format()` output contains the structured JSON
as a string (because the model's reasoning channel separates content
differently). The string looks like:
```json
{"sub_tasks": [{"sub_task": "...", "sub_task_agent": "researcher", ...}]}
```
The markdown parser `_parse_plan_string` finds no `"- Step"` lines in this
JSON string, so it returns an empty list. The result:
`final_plan.json` is written as `{"sub_tasks": []}`, and the control phase
crashes with `IndexError: list index out of range` when trying to read
`plan_input[0]['sub_task_agent']`.

Additionally, the JSON often contains un-escaped LaTeX backslashes (e.g.
`\frac`, `\sum`, `\ell`) because gpt-oss was trained on math-heavy data.
Standard `json.loads()` rejects these as invalid escape sequences.

#### How the patch works

```python
import sys, json, re
# Use sys.modules to get the actual module object.
# `import cmbagent.workflows.deep_research as X` would get the function
# (re-exported by __init__.py), not the module.
_dr_mod = sys.modules['cmbagent.workflows.deep_research']

# Save a reference to the original function BEFORE replacing it.
_orig_sfp = _dr_mod.save_final_plan

def _patched_sfp(final_context, work_dir):
    plan_obj = final_context.get("final_plan")

    # Only intervene when the plan is a string (the problematic case).
    # If it's already a Pydantic object or dict, skip straight to original.
    if isinstance(plan_obj, str):
        try:
            # Attempt 1: parse as standard JSON.
            try:
                parsed = json.loads(plan_obj)

            except json.JSONDecodeError:
                # Attempt 2: fix un-escaped LaTeX backslashes.
                # The regex replaces any `\X` that is NOT a valid JSON
                # escape (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`,
                # `\uXXXX`) with `\\X` (properly escaped).
                # Example: `\frac` → `\\frac`, `\sum` → `\\sum`
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', plan_obj)

                # strict=False allows control characters (newlines, tabs)
                # inside JSON string values, which gpt-oss sometimes emits
                # in LaTeX equations.
                parsed = json.JSONDecoder(strict=False).decode(fixed)

            # If we got a dict with "sub_tasks", replace the string with
            # the parsed dict so the original save_final_plan hits
            # "Case 2: already a dict" and writes it correctly.
            if isinstance(parsed, dict) and "sub_tasks" in parsed:
                final_context = dict(final_context)  # shallow copy
                final_context["final_plan"] = parsed

        except Exception:
            pass  # if all parsing fails, let the original handle it

    # Call the original function (now with a dict instead of a string).
    return _orig_sfp(final_context, work_dir)

# Sentinel attribute prevents double-patching if the cell is re-run.
# Without this, re-running would set _orig_sfp = _patched_sfp (the
# already-patched version), causing infinite recursion.
_patched_sfp._gptoss_patched = True

# Replace the function in the module's namespace. When deep_research()
# calls save_final_plan(), Python looks up the name in the function's
# __globals__ (which IS the module's __dict__), so it finds our version.
_dr_mod.save_final_plan = _patched_sfp
```

### Patch 2: `normilize_message_to_oai` (empty content / empty tool_calls)

#### The problem

When an AG2 agent receives an LLM response, it calls `speaker.send(reply,
...)` which internally calls `_append_oai_message(message, ...)`. This
function calls `normilize_message_to_oai(message, ...)` to validate the
message before appending it to the conversation history.

The validation logic (simplified):
```python
def normilize_message_to_oai(message, name, role):
    oai_message = {k: message[k] for k in (...) if k in message and message[k] is not None}
    if tools := message.get("tool_calls"):  # checks for [], None, missing
        oai_message["tool_calls"] = tools
    if "content" not in oai_message:
        if "function_call" in oai_message or "tool_calls" in oai_message:
            oai_message["content"] = None    # OK: tool call without content
        else:
            return False, oai_message        # INVALID: no content, no calls
```

With **GPT-4.1**: when no tool call is made, `tool_calls` is `None`. The
`if tools := message.get("tool_calls")` check is falsy for `None`, so
`tool_calls` is not added to `oai_message`. If `content` is present,
validation passes.

With **gpt-oss-120b**: several things go wrong:

1. **`tool_calls=[]` (empty list, not None)**. The OpenAI tool parser in
   vLLM returns `[]` when the model didn't invoke any function. Earlier in
   the pipeline, `extract_text_or_completion_object` checks
   `choice.message.tool_calls is not None` -- `[] is not None` is `True`,
   so it returns the full message object (instead of just the content
   string). After `model_dump()`, the dict has `"tool_calls": []`.
   In `normilize_message_to_oai`, `if tools := message.get("tool_calls")`
   is falsy for `[]`, so `tool_calls` is NOT added. But `content` might
   also be `None` (see below), causing validation failure.

2. **`content=None` or `content=""`**. When the model puts all its output
   into the reasoning/analysis channel (`<|channel|>analysis`), the final
   channel (`<|channel|>final`) is empty. vLLM sets `message.content = ""`
   or `None` and `message.reasoning_content = "the actual text..."`.
   AG2 filters out `None` content, so `oai_message` has neither `content`
   nor `tool_calls` → validation returns `False` → `send()` raises
   `ValueError`.

#### How the patch works

```python
import autogen.agentchat.conversable_agent as _ca_mod

# Save original before replacing.
_orig_normalize = _ca_mod.normilize_message_to_oai

def _patched_normalize(message, name, role="assistant"):
    if isinstance(message, dict):

        # Fix 1: Strip empty tool_calls=[].
        # This prevents downstream confusion where AG2 might treat it
        # as a tool-call response with no actual calls.
        tc = message.get("tool_calls")
        if isinstance(tc, list) and len(tc) == 0:
            message = {k: v for k, v in message.items() if k != "tool_calls"}

        # Fix 2: Fall back to reasoning_content.
        # If the model put all its output in the analysis channel and
        # left content empty, use the reasoning text as content so the
        # message is still valid in the conversation history.
        if not message.get("content") and message.get("reasoning_content"):
            message = dict(message)
            message["content"] = message["reasoning_content"]

        # Fix 3: Last resort placeholder.
        # If content is STILL empty and there are no tool calls or
        # function calls, set a placeholder to prevent the ValueError.
        # This should rarely trigger -- it's a safety net.
        if not message.get("content") and "function_call" not in message and not message.get("tool_calls"):
            message = dict(message)
            message["content"] = "(no response)"

    # Call the original validation with the cleaned-up message.
    return _orig_normalize(message, name, role)

# Sentinel to prevent double-patching (same pattern as Patch 1).
_patched_normalize._gptoss_patched = True

# Replace in the module namespace. normilize_message_to_oai is called
# by _append_oai_message which looks it up as a module-level function,
# so replacing it in the module dict is sufficient.
_ca_mod.normilize_message_to_oai = _patched_normalize
```

### Why monkey-patching (not modifying the source)

- **cmbagent** is in a shared repo (`/scratch/scratch-lxu/cmbagent_borisbolliet/`)
  that other users and workflows depend on. Modifying it would affect
  all models, not just gpt-oss.
- **AG2** is similarly shared. The `tool_calls=[]` vs `None` issue is
  specific to vLLM's OpenAI tool parser; OpenAI's own API always returns
  `None`.
- Monkey-patching in the notebook keeps the fixes isolated to the
  gpt-oss-120b use case without side effects.

### Why `sys.modules` instead of `import X as mod`

Python's `from cmbagent.workflows import deep_research` imports the
**function** `deep_research` from the package. If `__init__.py` does
`from .deep_research import deep_research`, then the attribute
`cmbagent.workflows.deep_research` on the package object is the function,
not the submodule. Using `import cmbagent.workflows.deep_research as X`
can resolve to either the module or the function depending on import
order. `sys.modules['cmbagent.workflows.deep_research']` always gives the
**module** object, which is what we need to replace its `save_final_plan`
attribute.

### Why the `_gptoss_patched` sentinel

Without it, re-running the patch cell does:
```
_orig_sfp = _dr_mod.save_final_plan  # this is ALREADY _patched_sfp!
def _patched_sfp(...):
    ...
    return _orig_sfp(...)  # calls itself → infinite recursion
```
The sentinel check (`if not getattr(func, '_gptoss_patched', False)`)
skips the patching if it was already applied.

---

## 5. Summary

| Feature | Works out of the box? | Notes |
|---------|----------------------|-------|
| Basic chat | Yes | Content in `message.content`, reasoning in `message.reasoning_content` |
| Structured output (Pydantic) | Yes | Guided decoding works; watch for LaTeX backslashes in JSON |
| Tool calling | Mostly | Empty `tool_calls=[]` and empty content need AG2 patch |
| CMBAgent one_shot | Yes | Works after AG2 patch |
| CMBAgent deep_research | Yes | Needs both patches (plan parsing + AG2 message handling) |
