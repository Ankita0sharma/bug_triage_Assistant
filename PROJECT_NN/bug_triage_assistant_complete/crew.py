import os
import re
import logging
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import litellm
from typing import Optional

# Load environment variables
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure GEMINI_API_KEY is available to litellm
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

def _first_fenced_code(text: str) -> Optional[str]:
    """Return the first fenced code block content (without fences) in the text, else None."""
    if not text:
        return None
    # Matches ```lang\n...\n``` or ```\n...\n```
    m = re.search(r"```(?:\w+)?\n([\s\S]*?)\n```", text)
    if m:
        return m.group(1).strip()
    # Also accept indented code blocks (4 spaces) as a fallback
    lines = text.splitlines()
    code_lines = [ln[4:] for ln in lines if ln.startswith("    ")]
    if code_lines:
        return "\n".join(code_lines).strip()
    return None

def _shorten(text: str, max_chars: int = 5000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated]"

class BugTriageCrew:
    def __init__(self):
        # use a current model (adjust to your environment if needed)
        self.model_name = "gemini/gemini-2.0-flash"

        # Agents
        self.log_agent = Agent(
            role="Log Analyzer",
            goal="Extract error type, file, line and key message from logs.",
            backstory="Expert at reading stack traces and error logs.",
            verbose=False,
            allow_delegation=False,
            llm=self.model_name,
        )

        self.root_agent = Agent(
            role="Root Cause Analyst",
            goal="Identify what went wrong, where, and why.",
            backstory="Senior debugger with deep problem solving skills.",
            verbose=False,
            allow_delegation=False,
            llm=self.model_name,
        )

        self.fix_agent = Agent(
            role="Fix Generator",
            goal="Provide step-by-step fixes and corrected code.",
            backstory="Mentor developer who explains fixes clearly and provides minimal patches.",
            verbose=False,
            allow_delegation=False,
            llm=self.model_name,
        )

    def _make_tasks(self, error_log: str, code_snippet: str):
        t1 = Task(
            description=f"""Extract error type, file, line number and short meaning from the error log.

ERROR LOG:
{error_log}
""",
            expected_output="""
A concise JSON-like summary with keys: error_type, file, line_number, short_message.
Example:
error_type: "TypeError"
file: "app.py"
line_number: 123
short_message: "NoneType is not callable"
""",
            agent=self.log_agent,
        )

        t2 = Task(
            description=f"""Using the error log and (optional) code, find the root cause.

ERROR LOG:
{error_log}

CODE:
{code_snippet}
""",
            expected_output="""
A short explanation that points to the source of the bug (file/line or function), why it occurs, and evidence from the log or code.
""",
            agent=self.root_agent,
        )

        t3 = Task(
            description=f"""Provide a step-by-step fix and corrected code. Keep the patch minimal: only include changed lines or a concise fixed file. Use fenced code blocks for any code.

ERROR LOG:
{error_log}

CODE:
{code_snippet}
""",
            expected_output="""
1) Short summary of the fix.
2) Fixed code block(s) fenced with triple-backticks (```).
3) Any verification steps/commands to test the fix.
""",
            agent=self.fix_agent,
        )

        return [t1, t2, t3]

    def run(self, error_log: str, code_snippet: Optional[str] = None) -> dict:
        """
        Run the three-agent crew and return a dict with keys:
        - log_analysis
        - root_cause
        - fix_suggestion
        - fix_code  (raw code if found, else "")
        """
        if not code_snippet:
            code_snippet = "No code snippet provided."

        tasks = self._make_tasks(error_log, code_snippet)
        crew = Crew(agents=[self.log_agent, self.root_agent, self.fix_agent], tasks=tasks, verbose=False)

        try:
            crew.kickoff()
        except Exception as exc:
            logger.exception("Error during crew.kickoff()")
            return {
                "log_analysis": f"ERROR during kickoff: {exc}",
                "root_cause": "",
                "fix_suggestion": "",
                "fix_code": ""
            }

        # Safely grab outputs
        try:
            out1 = tasks[0].output if hasattr(tasks[0], "output") else None
            out2 = tasks[1].output if hasattr(tasks[1], "output") else None
            out3 = tasks[2].output if hasattr(tasks[2], "output") else None

            # Convert to strings and shorten if extremely long
            s1 = _shorten(str(out1 or ""), max_chars=4000)
            s2 = _shorten(str(out2 or ""), max_chars=4000)
            s3 = _shorten(str(out3 or ""), max_chars=8000)  # fix text might be longer

            # Extract first fenced code block from fix agent output (if any)
            fix_code = _first_fenced_code(s3) or ""

            # Create a human-friendly fix suggestion text (remove the fenced block itself
            # if present, to avoid duplication)
            if fix_code:
                # remove the first fenced block from s3
                s3_no_code = re.sub(r"```(?:\w+)?\n[\s\S]*?\n```", "", s3, count=1).strip()
            else:
                s3_no_code = s3

            # Final result
            result = {
                "log_analysis": s1,
                "root_cause": s2,
                "fix_suggestion": s3_no_code,
                "fix_code": fix_code
            }

            # Ensure no None values
            for k in ["log_analysis", "root_cause", "fix_suggestion", "fix_code"]:
                if result.get(k) is None:
                    result[k] = ""

            logger.info("Crew run completed. Received outputs (lengths): %d, %d, %d",
                        len(result["log_analysis"]), len(result["root_cause"]), len(result["fix_suggestion"]))
            return result

        except Exception as exc:
            logger.exception("Error extracting task outputs")
            return {
                "log_analysis": "Error extracting outputs from crew tasks.",
                "root_cause": str(exc),
                "fix_suggestion": "",
                "fix_code": ""
            }
