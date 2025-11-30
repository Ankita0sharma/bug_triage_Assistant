from flask import Flask, render_template, request
from crew import BugTriageCrew

app = Flask(__name__)
crew = BugTriageCrew()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    error_log = request.form.get("error_log", "").strip()
    code_snippet = request.form.get("code_snippet", "").strip()

    if not error_log:
        result = {
            "log_analysis": "Please provide an error log.",
            "root_cause": "",
            "fix_suggestion": "",
        }
        return render_template("result.html", result=result)

    result = crew.run(error_log, code_snippet or None)
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
