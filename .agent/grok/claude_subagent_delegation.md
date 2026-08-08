---
### SUBAGENT DELEGATION PROTOCOL: CLAUDE

**Identity & Capability:**
You can deploy a Claude AI subagent via the local terminal using the `claude` CLI command. Claude is a stateless worker that excels at deep refactoring, nuanced code generation, and complex technical writing.

**When to Delegate:**
Invoke the Claude subagent for:
*   **Deep Refactoring:** Restructuring a React/TypeScript component tree in `src/components/` without changing rendered output.
*   **UI/Frontend Generation:** Generating structured code for complex layouts (e.g., a new content-listing or data-visualization component).
*   **Granular Code Review:** Performing rigorous audits of a notebook's analysis logic before it's cited in a report.

**Execution Syntax:**
Run the command in your shell, wrapping the prompt in strong quotes.
`claude 'YOUR_COMPREHENSIVE_PROMPT_HERE'`

**Subagent Prompting Rules (How to talk to Claude):**
1.  **Complete Independence:** Claude cannot read your memory. You MUST provide the exact code block or exact error logs it needs to act upon.
2.  **ReAct / CoT Triggers:** Instruct Claude to use `<thinking>` XML blocks to plan its refactoring steps before outputting code.
3.  **Strict Boundaries:** Specify exact input and output formats (e.g., "Output ONLY valid TypeScript code inside a single markdown block").

**Example Usage:**
`claude 'Act as an expert React/TypeScript developer. Refactor the following PostWrapper component so it shares its Markdown-rendering logic with ReportWrapper via a common hook, without changing rendered output. Wrap your reasoning in <thinking> tags, then provide the refactored code. Code to refactor: [INSERT_CODE_HERE]'`

**Failure Modes to Avoid:**
*   **Do not** include single quotes inside the prompt string without escaping them.
*   **Do not** delegate tasks that require multi-turn conversational context.
---