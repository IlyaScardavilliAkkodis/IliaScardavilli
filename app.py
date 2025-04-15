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

def analyze_cv_with_ai(pdf_path: str, job_description: str):
    try:
        reader = PdfReader(pdf_path)
        cv_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"

        if len(cv_text)>50:
            try:
                lang = detect(cv_text[:500])
                if not lang.startswith("it"):
                    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
                    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")
                    chunks = [cv_text[i:i+512] for i in range(0, len(cv_text), 512)]
                    translated = []
                    for ch in chunks:
                        inputs = tokenizer(ch, return_tensors="pt", truncation=True, padding=True)
                        tokens = model.generate(**inputs)
                        chunk_tradotto = tokenizer.decode(tokens[0], skip_special_tokens=True)
                        translated.append(chunk_tradotto)
                    cv_text = "\n".join(translated)
            except:
                pass

        res = role_classifier(cv_text[:512])
        role = res[0]['label']

        tskills, sskills = extract_skills(cv_text)
        cv_emb = sentence_model.encode(cv_text)
        job_emb = sentence_model.encode(job_description)
        sim = cosine_similarity([cv_emb],[job_emb])[0][0]
        adequacy = max(1, min(10, int(sim*10)))

        job_skills, _ = extract_skills(job_description)
        missing = [s for s in job_skills if s not in tskills]

        analysis = {
            "ruolo_principale": role,
            "competenze_tecniche": tskills,
            "competenze_trasversali": sskills,
            "adeguatezza": adequacy,
            "motivazione_adeguatezza": f"Punteggio {adequacy}/10",
            "competenze_mancanti": missing,
            "domande_colloquio": [],
            "email": None,
            "telefono": None,
            "localita": None,
            "eta": None,
            "disponibilita_mobilita": False,
            "esperienze_mobilita": [],
            "defense_compatibility": {}
        }

        return analysis

    except Exception as e:
        return {"error": str(e)}

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
        filename = secure_filename(file.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fpath)
        analysis = analyze_cv_with_ai(fpath, job_desc)
        os.remove(fpath)
        user = session["user"]
        if user not in USER_HISTORY:
            USER_HISTORY[user] = []
        USER_HISTORY[user].append({
            "cvName": filename,
            "analysis": analysis,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history_to_json()
        return jsonify(analysis)
    return jsonify({"error":"Invalid file type"}),400

# ------------------ HISTORY ------------------

@app.route("/history")
def history():
    if "user" not in session:
        return jsonify({"error": "Non sei loggato!"}), 403
    user = session["user"]
    if is_admin():
        full_hist = []
        for usr, hist_list in USER_HISTORY.items():
            for item in hist_list:
                full_hist.append({**item, "owner": usr})
        return jsonify(full_hist)
    else:
        user_list = USER_HISTORY.get(user, [])
        return jsonify([{**item, "owner": user} for item in user_list])

# ------------------ AVVIO ------------------

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)
