---
### SUBAGENT DELEGATION PROTOCOL: CLAUDE

**Identity & Capability:**
You are equipped with a CLI tool to orchestrate a Claude AI subagent via the `claude` command. Claude executes statelessly and has no awareness of this current chat session or your previous outputs.

**When to Delegate:**
Invoke the Claude subagent for:
*   **Deep Refactoring:** Restructuring a React/TypeScript component tree in `src/components/` without changing rendered output.
*   **UI/Frontend Generation:** Generating structured code for visually complex layouts (e.g., a new content-listing or data-visualization component).
*   **Code Review:** Performing rigorous, independent audits of a notebook's analysis logic before it's cited in a report.

**Execution Syntax:**
Execute the command in your terminal environment. Always enclose the prompt in single quotes to protect shell formatting.
`claude 'YOUR_COMPREHENSIVE_PROMPT_HERE'`

**Subagent Prompting Rules (How to talk to Claude):**
1.  **Context Injection:** You MUST paste all relevant snippets, constraints, and current state into the prompt. 
2.  **Strict Boundaries:** Clearly define what Claude should NOT do to save processing time (e.g., "Do not write explanations, output only the refactored component file").
3.  **Step-by-Step Prompting:** For complex tasks, instruct Claude to use a `<thinking>` XML block before providing the final answer.

**Example Usage:**
`claude 'You are an expert TypeScript/React developer. Below is the shape of the front-matter for a report under app/content/reports/. Write a well-typed TS interface for it, and generate a generic React component to render its metadata card. Wrap your reasoning in <thought> tags and output the code in a single markdown block. Context: [INSERT_FRONTMATTER_SHAPE_HERE]'`

**Failure Modes to Avoid:**
*   **Do not** nest quotes improperly (e.g., `claude 'He said 'hello''`). 
*   **Do not** ask Claude to perform actions it cannot do (like interacting with your local file system directly). You must parse its text output and perform the file operations yourself.
*   **Do not** use ambiguous instructions; quantify your requests (e.g., "Provide exactly 2 solutions").
---