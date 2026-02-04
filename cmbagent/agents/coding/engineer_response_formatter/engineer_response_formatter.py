import os
from cmbagent.base_agent import BaseAgent
from pydantic import BaseModel, Field
from typing import Optional
class EngineerResponseFormatterAgent(BaseAgent):
    
    def __init__(self, llm_config=None, **kwargs):

        agent_id = os.path.splitext(os.path.abspath(__file__))[0]

        llm_config['config_list'][0]['response_format'] = self.EngineerResponse
        # llm_config['config_list'][0]['response_mime_type'] = "application/json"
        # llm_config['config_list'][0]['response_schema'] = list[self.EngineerResponse]

        super().__init__(llm_config=llm_config, agent_id=agent_id, **kwargs)


    def set_agent(self,**kwargs):

        super().set_assistant_agent(**kwargs)



    class EngineerResponse(BaseModel):
        filename: str = Field(..., description="The name to give to this Python script")
        relative_path: Optional[str] = Field(
            None, description="The relative path to the file (exclude <filename>.py itself)"
        )
        code_explanation: str = Field(
            ..., description="Copy of the engineer's explanation of the Python code provided. Including the docstrings of the methods used."
        )
        modification_summary: Optional[str] = Field(
            None,
            description="Copy of the engineer's summary of any modifications made to fix errors from the previous version."
        )
        python_code: str = Field(
            ..., description="Copy of the engineer's Python code in a form ready to execute. Should not contain anything else than code. Indentation has to be carefully checked, line by line, and fixed."
        )

        @staticmethod
        def _fix_indentation(code: str, max_attempts: int = 10) -> str:
            """Auto-fix indentation errors caused by LLM structured output.

            When an LLM generates Python code inside a JSON string field
            (e.g., the ``python_code`` field of a Pydantic structured response),
            it occasionally introduces stray whitespace — typically one extra
            space at the start of a line. This breaks Python's indentation
            rules and causes ``IndentationError`` at execution time.

            This method uses ``compile()`` to detect such errors and attempts
            to repair them automatically. For each offending line reported by
            the compiler, it builds a set of candidate indentation levels from
            the surrounding context:

              - **Same level** as the nearest non-empty line above.
              - **One level deeper** if that line opens a block (ends with ``:``,
                e.g., ``for``, ``if``, ``def``, ``while``, ``with``, ...).
              - **One level shallower** (dedent) — for lines that follow the
                end of a block body and return to the parent scope.
              - **Same level** as the nearest non-empty line below.
              - **Zero indent** (module level) as a fallback.

            Each candidate is tried with ``compile()``; the first one that
            produces valid Python wins. The process repeats up to
            ``max_attempts`` times to handle multiple affected lines.

            Parameters
            ----------
            code : str
                The full Python source code to check and fix.
            max_attempts : int, optional
                Maximum number of lines to fix in one pass (default 10).

            Returns
            -------
            str
                The code with indentation errors fixed where possible.
                If the code has non-indentation ``SyntaxError``s, or if no
                candidate fixes compile, the code is returned as-is.
            """
            print("[_fix_indentation] Checking code for indentation errors...")
            for attempt in range(max_attempts):
                try:
                    compile(code, "<string>", "exec")
                    if attempt == 0:
                        print("[_fix_indentation] Code compiles cleanly — no fixes needed.")
                    else:
                        print("[_fix_indentation] Code now compiles after " + str(attempt) + " fix(es).")
                    return code
                except IndentationError as exc:
                    if exc.lineno is None:
                        print("[_fix_indentation] IndentationError with no line number — cannot fix.")
                        return code
                    lines = code.splitlines()
                    bad_idx = exc.lineno - 1
                    if bad_idx < 0 or bad_idx >= len(lines):
                        print("[_fix_indentation] Bad line index out of range — cannot fix.")
                        return code

                    bad_content = lines[bad_idx].lstrip()
                    if not bad_content:
                        print("[_fix_indentation] Offending line is empty — cannot fix.")
                        return code

                    print("[_fix_indentation] IndentationError on line " + str(exc.lineno) + ": " + repr(lines[bad_idx]))

                    # Collect candidate indentation levels from context
                    candidates = []

                    # From nearest non-empty neighbour above
                    for i in range(bad_idx - 1, -1, -1):
                        above = lines[i]
                        stripped = above.lstrip()
                        if stripped and not stripped.startswith("#"):
                            above_indent = above[: len(above) - len(stripped)]
                            # Same level as neighbour above
                            candidates.append(above_indent)
                            # One level deeper (if neighbour ends with ':')
                            if stripped.rstrip().endswith(":"):
                                candidates.append(above_indent + "    ")
                            # One level shallower (dedent)
                            if len(above_indent) >= 4:
                                candidates.append(above_indent[:-4])
                            break

                    # From nearest non-empty neighbour below
                    for i in range(bad_idx + 1, len(lines)):
                        below = lines[i]
                        stripped = below.lstrip()
                        if stripped and not stripped.startswith("#"):
                            below_indent = below[: len(below) - len(stripped)]
                            candidates.append(below_indent)
                            break

                    # Always try zero indent (module level)
                    candidates.append("")

                    # Deduplicate while preserving order
                    seen = set()
                    unique_candidates = []
                    for c in candidates:
                        if c not in seen:
                            seen.add(c)
                            unique_candidates.append(c)

                    print("[_fix_indentation] Trying " + str(len(unique_candidates)) + " candidate indent(s): " + str([str(len(c)) + " spaces" for c in unique_candidates]))

                    # Try each candidate — use the first one that compiles
                    fixed = False
                    for candidate in unique_candidates:
                        trial_lines = lines[:]
                        trial_lines[bad_idx] = candidate + bad_content
                        trial_code = "\n".join(trial_lines)
                        try:
                            compile(trial_code, "<string>", "exec")
                            code = trial_code
                            fixed = True
                            print("[_fix_indentation] Fixed line " + str(exc.lineno) + " -> " + str(len(candidate)) + " spaces indent: " + repr(candidate + bad_content))
                            break
                        except SyntaxError:
                            continue

                    if not fixed:
                        # None of the candidates compiled on their own, but
                        # picking the best guess still lets the next iteration
                        # fix a subsequent line.  Prefer the neighbour-below
                        # indent if available, otherwise the first candidate.
                        best = unique_candidates[-2] if len(unique_candidates) >= 2 else unique_candidates[0]
                        lines[bad_idx] = best + bad_content
                        code = "\n".join(lines)
                        print("[_fix_indentation] No candidate compiled — using best guess (" + str(len(best)) + " spaces) for line " + str(exc.lineno))

                except SyntaxError:
                    # Not an indentation issue — nothing we can auto-fix
                    print("[_fix_indentation] Non-indentation SyntaxError — returning code as-is.")
                    return code
            print("[_fix_indentation] Reached max attempts (" + str(max_attempts) + ") — returning best effort.")
            return code

        def format(self) -> str:
            final_filename = self.filename if self.filename.endswith(".py") else self.filename + ".py"

            if self.relative_path:
                cleaned_path = self.relative_path.rstrip("/\\")
                full_path = os.path.join(cleaned_path, os.path.basename(final_filename))
            else:
                full_path = final_filename

            # Preamble: filename comment + sys.path for codebase imports
            preamble_lines = [
                f"# filename: {full_path}",
                "import sys",
                "import os",
                'sys.path.insert(0, os.path.abspath("codebase"))',
            ]

            code_lines = self.python_code.splitlines()

            # Strip any existing preamble the LLM may have included
            while code_lines and code_lines[0].strip() in (
                "", "import sys", "import os",
                'sys.path.insert(0, os.path.abspath("codebase"))',
                "sys.path.insert(0, os.path.abspath('codebase'))",
            ) or (code_lines and code_lines[0].strip().startswith("# filename:")):
                code_lines.pop(0)

            updated_python_code = "\n".join(preamble_lines + code_lines)

            # Fix indentation errors introduced by LLM structured output
            updated_python_code = self._fix_indentation(updated_python_code)

            response_parts = [f"**Code Explanation:**\n\n{self.code_explanation}"]

            if self.modification_summary:
                response_parts.append(
                    f"**Modifications:**\n\n{self.modification_summary}"
                )

            response_parts.append(
                f"**Python Code:**\n\n```python\n{updated_python_code}\n```"
            )

            return "\n\n".join(response_parts)


