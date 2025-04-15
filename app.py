import os
import re
import base64
import io
import json
import pdfplumber
from PIL import Image
from PyPDF2 import PdfReader
from langdetect import detect
import numpy as np
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename

from transformers import pipeline, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ GESTIONE UTENTI ------------------
USER_DB = {
    "admin": "adminpass",
    "Ilia": "password1",
    "Alfonso": "password2",
    "Giordano": "password3",
    "Noemi": "password4"
}

USER_HISTORY = {}
HISTORY_FILE = "user_history.json"

def load_history_from_json():
    global USER_HISTORY
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            USER_HISTORY = json.load(f)
    else:
        USER_HISTORY = {}

def save_history_to_json():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(USER_HISTORY, f, ensure_ascii=False, indent=2)

# ------------------ FLASK APP ------------------

app = Flask(__name__)
app.secret_key = "S3gretoMoltoTemp0raneo"

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Loading AI models...")
load_history_from_json()

# ------------------ MODELLI ------------------

role_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

TECHNICAL_SKILLS = {
    "programming": ["python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust", "swift"],
    "web": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "spring", "laravel"],
    "database": ["sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite"],
    "devops": ["docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "git", "ci/cd"],
    "ai_ml": ["tensorflow", "pytorch", "scikit-learn", "numpy", "pandas", "machine learning", "deep learning"],
    "mobile": ["android", "ios", "react native", "flutter", "kotlin", "swift"],
    "cloud": ["aws", "azure", "google cloud", "serverless", "lambda", "ec2", "s3"],
    "security": ["cybersecurity", "penetration testing", "encryption", "authentication", "authorization"]
}

SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving", "time management",
    "adaptability", "creativity", "critical thinking", "emotional intelligence",
    "conflict resolution", "negotiation", "decision making", "stress management"
]

# ------------------ LOGIN ------------------

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username in USER_DB and USER_DB[username] == password:
            session["user"] = username
            if username not in USER_HISTORY:
                USER_HISTORY[username] = []
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Credenziali non valide!")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

def is_admin():
    return "user" in session and session["user"] == "admin"

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["user"])

# ------------------ ANALISI AI ------------------
# ... (analisi functions invariati)

# ------------------ UPLOAD ------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "user" not in session:
        return jsonify({"error": "Non sei loggato"}), 403
    if 'file' not in request.files:
        return jsonify({"error":"No file part"}),400
    file = request.files['file']
    job_desc = request.form.get('job_description',"")
    if file.filename == "":
        return jsonify({"error":"No selected file"}),400
    if file and file.filename.lower().endswith(".pdf"):
        try:
            filename = secure_filename(file.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fpath)
            analysis = analyze_cv_with_ai(fpath, job_desc)
            os.remove(fpath)
            user = session["user"]
            USER_HISTORY.setdefault(user, []).append({
                "cvName": filename,
                "analysis": analysis,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_history_to_json()
            if not isinstance(analysis, dict):
                return jsonify({"error": "Errore ignoto nell'analisi"}), 500
            if "error" in analysis:
                return jsonify(analysis), 500
            return jsonify(analysis)
        except Exception as e:
            return jsonify({"error": f"Errore server: {str(e)}"}), 500
    return jsonify({"error":"Invalid file type"}),400

# ------------------ HISTORY ------------------

@app.route("/history")
def history():
    if "user" not in session:
        return jsonify({"error": "Non sei loggato!"}), 403
    user = session["user"]
    if is_admin():
        return jsonify([
            {**item, "owner": usr}
            for usr, hist_list in USER_HISTORY.items()
            for item in hist_list
        ])
    else:
        return jsonify([{**item, "owner": user} for item in USER_HISTORY.get(user, [])])

# ------------------ FINTA ROTTA /extract_photo (per evitare errori 404) ------------------
@app.route("/extract_photo", methods=["POST"])
def extract_photo():
    return jsonify({"photo": None, "name": None})

# ------------------ AVVIO ------------------

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
