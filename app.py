import os
import sys

from flask import Flask, jsonify, render_template, request
from markdown import markdown

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from main import run_news_crew
from audio import generate_audio_summary

app = Flask(__name__)


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    if "timed out" in lower or "timeout" in lower:
        return "The agent workflow took too long to get a response from the model. Please try again, or use a slightly more specific topic."
    if "rate limit" in lower or "429" in lower:
        return "The AI model rate limit was reached. Wait a few seconds and try again."
    if "ratelimit" in lower or "202" in lower:
        return "The search engine is temporarily blocking your IP for too many requests. Please wait 5 minutes."
    return message


@app.route("/", methods=["GET", "POST"])
def index():
    topic = ""
    result = None
    error = None

    if request.method == "POST":
        topic = (request.form.get("topic") or "").strip()
        audio_option = (request.form.get("audio_option") or "brief").strip()
        if not topic:
            error = "Please enter a topic."
        else:
            try:
                result = run_news_crew(topic)
                
                try:
                    audio_file = generate_audio_summary(
                        topic, 
                        result.get("research_overview", ""),
                        result.get("findings", []),
                        audio_option,
                        result.get("recommended_angle", "")
                    )
                    result["audio_filename"] = audio_file
                except Exception as e:
                    print(f"Audio generation failed: {e}")

                result["report_html"] = markdown(
                    result["report"],
                    extensions=["extra", "sane_lists", "tables"],
                )
                return render_template("report.html", topic=topic, result=result)
            except Exception as exc:
                error = _friendly_error(exc)

    return render_template("index.html", topic=topic, error=error)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    topic = (payload.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    try:
        result = run_news_crew(topic)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": _friendly_error(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
