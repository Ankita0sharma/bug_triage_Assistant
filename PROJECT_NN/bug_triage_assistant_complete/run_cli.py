from crew import BugTriageCrew

triage = BugTriageCrew()

error_log = input("Paste your error log:\n")
code_snippet = input("Paste your code snippet (optional):\n")

result = triage.run(error_log, code_snippet)

print("\n--- Log Analysis ---")
print(result["log_analysis"])

print("\n--- Root Cause ---")
print(result["root_cause"])

print("\n--- Fix Suggestion ---")
print(result["fix_suggestion"])