import os
import time
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, send_file

from yoselog_core import YogaSessionEngine


app = Flask(__name__)
engine = YogaSessionEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_session():
    engine.start()
    return jsonify(engine.snapshot())


@app.route("/api/stop", methods=["POST"])
def stop_session():
    engine.stop()
    return jsonify(engine.snapshot())


@app.route("/api/reset", methods=["POST"])
def reset_session():
    engine.reset()
    return jsonify(engine.snapshot())


@app.route("/api/session")
def session_state():
    return jsonify(engine.snapshot())


@app.route("/api/export")
def export_session():
    os.makedirs("exports", exist_ok=True)
    filename = f"yoga_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join("exports", filename)
    engine.export_csv(path)
    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = engine.frame_bytes()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, threaded=True, use_reloader=False)
