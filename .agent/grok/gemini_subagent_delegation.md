---
### SUBAGENT DELEGATION PROTOCOL: GEMINI

**Identity & Capability:**
You have the authority to spawn a Gemini AI subagent via the terminal using the `agy` CLI command. Gemini operates independently, statelessly, and processes large contexts with high efficiency.

**When to Delegate:**
Invoke the Gemini subagent for:
*   **Data Wrangling:** Standardizing a notebook's data transformations or parsing large JSON/log/CSV inputs for a report.
*   **Front-Matter/Content Boilerplate:** Generating consistent Markdown front-matter and stub pages when adding a new `app/content/<section>/` entry.
*   **Long-Context Extraction:** Pulling structured summaries or key figures out of a long research write-up under `docs/research/`.

**Execution Syntax:**
Execute the command in your terminal. Ensure the prompt is enclosed in single quotes.
`agy 'YOUR_COMPREHENSIVE_PROMPT_HERE'`

**Subagent Prompting Rules (How to talk to Gemini):**
1.  **Explicit Context:** Provide all required schemas, data samples, and environmental constraints (e.g., Linux, KDE, specific GPU hardware).
2.  **Template Pattern:** Dictate the exact output structure using a template to ensure the response can be easily parsed or piped into another tool.
3.  **Action-Oriented Verbs:** Start instructions with clear directives like "Analyze," "Generate," or "Extract."

**Example Usage:**
`agy 'Act as an expert technical writer. Given the following notebook analysis output (summary stats + a chart description), draft the "Results" section of a Markdown report for app/content/reports/. Constraints: 1. Match the tone of the site's existing reports. 2. Output only the Markdown section. Context: [INSERT_ANALYSIS_OUTPUT]'`

**Failure Modes to Avoid:**
*   **Do not** use unescaped single quotes in the `agy` execution string.
*   **Do not** expect Gemini to read files from the disk automatically unless you ask it to generate the shell commands to do so.
---