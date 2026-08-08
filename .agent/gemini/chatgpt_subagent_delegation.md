---
### SUBAGENT DELEGATION PROTOCOL: CHATGPT

**Identity & Capability:**
You are equipped to launch a ChatGPT AI subagent via the `chatgpt` command. ChatGPT executes statelessly and has no awareness of this current chat session.

**When to Delegate:**
Invoke the ChatGPT subagent for:
*   **Content Drafting:** Drafting or tightening prose for a blog post/report before it's polished and placed under `app/content/`.
*   **Documentation & Abstraction:** Generating clear, high-level summaries of a notebook's analysis for the write-up that cites it.
*   **Alternative Paradigms:** Asking for a completely different framing of a post/report when the current draft feels stuck.

**Execution Syntax:**
Execute the command in your terminal environment. Always enclose the prompt in single quotes.
`chatgpt 'YOUR_COMPREHENSIVE_PROMPT_HERE'`

**Subagent Prompting Rules (How to talk to ChatGPT):**
1.  **Context Injection:** Paste all relevant snippets and constraints into the prompt.
2.  **Structured Output:** Use the Template Pattern. Define exactly how the output should look using a mock structure.
3.  **Chain-of-Thought:** For complex logic, explicitly ask ChatGPT to "Think step-by-step before providing the final answer."

**Example Usage:**
`chatgpt 'Act as a technical editor. Below is a draft section of a blog report on audio signal processing. Think step-by-step about where the explanation loses a non-specialist reader before tightening it. Output format:
## Reasoning: [Step-by-step thoughts]
## Revised Section: [Rewritten Markdown]'`

**Failure Modes to Avoid:**
*   **Do not** nest quotes improperly (e.g., `chatgpt 'He said 'hello''`).
*   **Do not** use ambiguous instructions; be explicit about the domain.
---