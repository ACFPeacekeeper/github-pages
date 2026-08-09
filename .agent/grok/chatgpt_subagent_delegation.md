---
### SUBAGENT DELEGATION PROTOCOL: CHATGPT

**Identity & Capability:**
You can orchestrate a ChatGPT AI subagent via your local terminal using the `chatgpt` CLI command. ChatGPT acts as a stateless, highly capable reasoning engine. It does not share your context window.

**When to Delegate:**
Invoke the ChatGPT subagent for:
*   **Mathematical Formulations:** Drafting formal definitions or notation for a report's math section (e.g. the PCVRP report under `app/content/reports/`).
*   **Literature & Concept Mapping:** Summarizing background material or related work to cite in a report.
*   **Creative Brainstorming:** Generating varied framings for a post/report before committing to a structure.

**Execution Syntax:**
Run the command in your shell, wrapping the prompt in strong quotes to prevent shell evaluation errors.
`chatgpt 'YOUR_COMPREHENSIVE_PROMPT_HERE'`

**Subagent Prompting Rules (How to talk to ChatGPT):**
1.  **Zero-Shot Context:** You MUST include all necessary definitions, constraints, and current state.
2.  **Constraint Pattern:** Explicitly list what ChatGPT must *not* do to keep the response focused and token-efficient.
3.  **Role Definition:** Always assign ChatGPT a clear persona (e.g., "Act as a PhD-level Operations Research scientist").

**Example Usage:**
`chatgpt 'Act as an Operations Research expert. I am writing a report on the Periodic Capacitated Vehicle Routing Problem (PCVRP). Provide the formal mathematical formulation for the objective function minimizing total travel cost over a multi-day horizon. Rules: 1. Use standard OR notation. 2. Define all variables clearly. 3. Output ONLY the formulation and variable definitions in plain text.'`

**Failure Modes to Avoid:**
*   **Do not** use unescaped single quotes inside the `chatgpt` command string.
*   **Do not** assume ChatGPT knows our current project state.
*   **Do not** delegate tasks requiring direct file manipulation.
---