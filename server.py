from flask import Flask, render_template
import subprocess
import sys
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable   # ensures same venv python

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/secure_in")
def secure_in():
    subprocess.call(
        [PYTHON_EXE, os.path.join(BASE_DIR, "main.py"), "IN"],
        cwd=BASE_DIR
    )
    return "IN attendance finished."

@app.route("/secure_out")
def secure_out():
    subprocess.call(
        [PYTHON_EXE, os.path.join(BASE_DIR, "main.py"), "OUT"],
        cwd=BASE_DIR
    )
    return "OUT attendance finished."

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
