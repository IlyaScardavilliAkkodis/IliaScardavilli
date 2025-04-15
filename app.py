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

########################################################################
#               Gestione UTENTI e PERSISTENZA CRONOLOGIE
########################################################################

# A) Dizionario utenti => password
USER_DB = {
    "admin": "adminpass",    # ADMIN può vedere tutto
    "ilia": "password1",
    "Alfonso": "password2",
    "Giordano": "password3",
    "Noemi": "password4"
}

# B) user_history (username -> [ { "cvName", "analysis", "timestamp" }, ... ])
#    Però lo salveremo e caricheremo da user_history.json
USER_HISTORY = {}  # verrà caricato da file

HISTORY_FILE = "user_history.json"

def load_history_from_json():
    """Carica la cronologia da un file JSON, se esiste."""
    global USER_HISTORY
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            USER_HISTORY = json.load(f)
    else:
        USER_HISTORY = {}

def save_history_to_json():
    """Salva l'intero dizionario USER_HISTORY su file JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(USER_HISTORY, f, ensure_ascii=False, indent=2)

########################################################################
#                      APP FLASK
########################################################################

app = Flask(__name__)
app.secret_key = "S3gretoMoltoTemp0raneo"

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Loading AI models...")

# Carichiamo la cronologia dal file
load_history_from_json()

########################################################################
#                 Modelli, skill, pipeline
########################################################################

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

########################################################################
#                         LOGIN/LOGOUT
########################################################################

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username in USER_DB and USER_DB[username] == password:
            session["user"] = username
            # inizializza se manca
            if username not in USER_HISTORY:
                USER_HISTORY[username] = []
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Credenziali non valide!")
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

def is_admin():
    return "user" in session and session["user"] == "admin"

########################################################################
#                         HOME
########################################################################

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    # username = session["user"]  # Se devi passarlo all'html
    return render_template("index.html", username=session["user"])

########################################################################
#                       FUNZIONI ANALISI
########################################################################

def translate_to_italian(text: str) -> str:
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    translated = []
    for ch in chunks:
        inputs = tokenizer(ch, return_tensors="pt", truncation=True, padding=True)
        tokens = model.generate(**inputs)
        chunk_tradotto = tokenizer.decode(tokens[0], skip_special_tokens=True)
        translated.append(chunk_tradotto)
    return "\n".join(translated)

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except:
        return ""

def analyze_role(text: str) -> str:
    res = role_classifier(text[:512])
    return res[0]['label']

def extract_skills(text: str):
    techset = set()
    softset = set()
    for cat, arr in TECHNICAL_SKILLS.items():
        for skill in arr:
            if skill.lower() in text.lower():
                techset.add(skill)
    for s in SOFT_SKILLS:
        if s.lower() in text.lower():
            softset.add(s)
    return list(techset), list(softset)

def calculate_adequacy(cv_text: str, job_desc: str) -> int:
    cv_emb = sentence_model.encode(cv_text)
    job_emb = sentence_model.encode(job_desc)
    sim = cosine_similarity([cv_emb],[job_emb])[0][0]
    sc = int(sim*10)
    return max(1, min(10, sc))

def generate_interview_questions(tech, soft, missing):
    qs = []
    for sk in tech:
        qs.append(f"Puoi descrivere un progetto in cui hai utilizzato {sk}?")
        qs.append(f"Quali best practices usi quando lavori con {sk}?")
    for s in soft:
        qs.append(f"Racconta un esempio di come hai dimostrato {s} in un progetto.")
    for m in missing:
        qs.append(f"Hai esperienza con {m}? Se sì, descrivila.")
    return qs

def extract_contact_info(text: str):
    cinfo = {
        'email': None,
        'telefono': None,
        'localita': None,
        'eta': None
    }
    # Email
    if m := re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text):
        cinfo['email'] = m.group(0)
    # Telefono
    if m := re.search(r'(?:\+39\s?)?\d{3}[\s-]?\d{3}[\s-]?\d{4}', text):
        cinfo['telefono'] = m.group(0)
    # Località
    if m := re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text):
        cinfo['localita'] = m.group(0)
    # Nascita => età (semplif.)
    birth_pat = r'(?:nato|nata)\s+(?:il\s+)?(\d{1,2}/\d{1,2}/\d{4})'
    if m := re.search(birth_pat, text, re.IGNORECASE):
        try:
            from datetime import datetime
            bd = datetime.strptime(m.group(1), '%d/%m/%Y')
            age = (datetime.now() - bd).days // 365
            cinfo['eta'] = f"{age} anni"
        except:
            pass
    return cinfo

def analyze_mobility(text: str):
    m = {
        'disponibilita_mobilita': False,
        'esperienze_mobilita': []
    }
    kw = [
        'disponibile a trasferte','disponibile a trasferimento','disponibile a spostamenti',
        'disponibile a viaggiare','disponibile a trasferirsi','disponibile a traslocare'
    ]
    for k in kw:
        if k in text.lower():
            m['disponibilita_mobilita'] = True
            break
    pat = r'(?:presso|a|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    found = re.findall(pat, text)
    skip = {'presso','a','in','il','la','lo','i','gli','le'}
    uniq = set(x for x in found if x.lower() not in skip)
    if uniq:
        m['esperienze_mobilita'] = list(uniq)
        if len(uniq)>1:
            m['disponibilita_mobilita'] = True
    return m

def extract_last_experience(text: str):
    return {
        'azienda': None,
        'periodo': None,
        'ruolo': None,
        'descrizione': None,
        'responsabilita': [],
        'attualmente_impiegato': False
    }

def analyze_defense_compatibility(text: str):
    analysis = {
        'is_defense_compatible': False,
        'explicit_availability': False,
        'relevant_experience': False,
        'security_clearance': False,
        'matching_keywords': [],
        'explanation': ''
    }
    explicit = [
        'disponibile a lavorare nel settore difesa','disponibile per progetti di difesa',
        'esperienza in ambito difesa','defense sector experience','military projects'
    ]
    project = ['progetto militare','progetto difesa','military project','defense project']
    clearance = ['security clearance','top secret','secret','confidential']
    for e in explicit:
        if e.lower() in text.lower():
            analysis['explicit_availability'] = True
            analysis['matching_keywords'].append(e)
    for p in project:
        if p.lower() in text.lower():
            analysis['relevant_experience'] = True
            analysis['matching_keywords'].append(p)
    for c in clearance:
        if c.lower() in text.lower():
            analysis['security_clearance'] = True
            analysis['matching_keywords'].append(c)
    analysis['is_defense_compatible'] = (
        analysis['explicit_availability'] or
        analysis['relevant_experience'] or
        analysis['security_clearance']
    )
    if analysis['is_defense_compatible']:
        reasons = []
        if analysis['explicit_availability']:
            reasons.append("ha espresso disponibilità esplicita")
        if analysis['relevant_experience']:
            reasons.append("ha esperienza in progetti rilevanti")
        if analysis['security_clearance']:
            reasons.append("ha autorizzazioni di sicurezza")
        analysis['explanation'] = "Il candidato è compatibile con il settore difesa perché " + " e ".join(reasons) + "."
    else:
        analysis['explanation'] = "Non emergono indicazioni di compatibilità difesa."
    return analysis

########################################################################
#              analizza_cv + analizza_cv_with_ai
########################################################################

def analizza_cv(pdf_path: str):
    # se preferisci pdfplumber:
    try:
        text = extract_text_from_pdf(pdf_path)
        # parse basic info
        # nel tuo codice c'era estrai_info, ecc. (semplifico)
        return {}
    except:
        return {}

def analyze_cv_with_ai(pdf_path: str, job_description: str):
    try:
        cv_text = extract_text_from_pdf(pdf_path)
        # detect language
        if len(cv_text)>50:
            try:
                lang = detect(cv_text[:500])
                if lang.startswith("it"):
                    pass
                else:
                    cv_text = translate_to_italian(cv_text)
            except:
                # euristica
                pass
        # cont + skill
        role = analyze_role(cv_text)
        tskills, sskills = extract_skills(cv_text)
        adequacy = calculate_adequacy(cv_text, job_description)
        job_skills, _ = extract_skills(job_description)
        missing = [s for s in job_skills if s not in tskills]
        cinfo = extract_contact_info(cv_text)
        mob = analyze_mobility(cv_text)
        lastExp = extract_last_experience(cv_text)
        defComp = analyze_defense_compatibility(cv_text)
        qs = generate_interview_questions(tskills, sskills, missing)

        return {
            "ruolo_principale": role,
            "competenze_tecniche": tskills,
            "competenze_trasversali": sskills,
            "adeguatezza": adequacy,
            "motivazione_adeguatezza": f"Punteggio {adequacy}/10",
            "competenze_mancanti": missing,
            "domande_colloquio": qs,
            "email": cinfo.get("email"),
            "telefono": cinfo.get("telefono"),
            "localita": cinfo.get("localita"),
            "eta": cinfo.get("eta"),
            "disponibilita_mobilita": mob["disponibilita_mobilita"],
            "esperienze_mobilita": mob["esperienze_mobilita"],
            "ultima_esperienza": lastExp,
            "defense_compatibility": defComp,
            "data_nascita": None,
            "sesso": None,
            "nazionalita": None,
            "indirizzo": None,
            "linkedin": None,
            "nome": None
        }
    except Exception as e:
        return {"error": str(e)}

########################################################################
#                 /upload => analizza e salva cronologia
########################################################################

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "user" not in session:
        return jsonify({"error": "Non sei loggato"}), 403

    if 'file' not in request.files:
        return jsonify({"error":"No file part"}),400

    file = request.files['file']
    job_desc = request.form.get('job_description',"")
    if file.filename=="":
        return jsonify({"error":"No selected file"}),400
    if file and file.filename.lower().endswith(".pdf"):
        filename = secure_filename(file.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fpath)

        analysis = analyze_cv_with_ai(fpath, job_desc)

        os.remove(fpath)

        # Salva in cronologia
        user = session["user"]
        if user not in USER_HISTORY:
            USER_HISTORY[user] = []
        # Salviamo con data/ora
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        USER_HISTORY[user].append({
            "cvName": filename,
            "analysis": analysis,
            "timestamp": timestamp
        })
        # Salva su disco
        save_history_to_json()

        return jsonify(analysis)

    return jsonify({"error":"Invalid file type"}),400

########################################################################
#        Rotta /history => se admin => tutti, altrimenti => solo user
########################################################################


@app.route("/history")
def history():
    if "user" not in session:
        return jsonify({"error": "Non sei loggato!"}), 403
    
    user = session["user"]
    
    if is_admin():
        # admin => unisci TUTTE le cronologie
        full_hist = []
        for usr, hist_list in USER_HISTORY.items():
            for item in hist_list:
                # item contiene { "cvName", "analysis", "timestamp", ... }
                # aggiungiamo "owner": usr
                full_hist.append({
                    "owner": usr,
                    "cvName": item["cvName"],
                    "analysis": item["analysis"],
                    "timestamp": item["timestamp"]
                })
        return jsonify(full_hist)
    else:
        # non admin => solo la cronologia di quell'utente
        user_list = USER_HISTORY.get(user, [])
        history_with_owner = []
        for item in user_list:
            # Copiamo item e aggiungiamo "owner" = user
            new_item = dict(item)
            new_item["owner"] = user
            history_with_owner.append(new_item)
        return jsonify(history_with_owner)

########################################################################
#    Estrarre foto
########################################################################

@app.route("/extract_photo", methods=["POST"])
def extract_photo():
    if "user" not in session:
        return jsonify({"error": "Non sei loggato"}),403

    if 'file' not in request.files:
        return jsonify({"error":"No file part"}),400

    file = request.files['file']
    if file.filename=="":
        return jsonify({"error":"No selected file"}),400

    if file and file.filename.lower().endswith(".pdf"):
        try:
            fname = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(path)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t+"\n"
            name = text.split("\n")[0].strip()

            photo_data = None
            for page in reader.pages:
                if '/XObject' in page['/Resources']:
                    xObj = page['/Resources']['/XObject'].get_object()
                    for obj in xObj:
                        if xObj[obj]['/Subtype'] == '/Image':
                            img = xObj[obj]
                            i_data = img._data
                            i_stream = io.BytesIO(i_data)
                            pil = Image.open(i_stream)
                            if pil.width<100 or pil.height<100:
                                continue
                            if pil.format not in ['JPEG','PNG']:
                                continue
                            buff = io.BytesIO()
                            pil.save(buff, format="PNG")
                            img_str = base64.b64encode(buff.getvalue()).decode()
                            photo_data = f"data:image/png;base64,{img_str}"
                            break
                if photo_data:
                    break
            os.remove(path)
            return jsonify({"photo":photo_data,"name":name})
        except Exception as e:
            return jsonify({"error":str(e)}),500
    return jsonify({"error":"Invalid file type"}),400

########################################################################
#   Test /analyze => se admin => etc. (semplice)
########################################################################

@app.route("/analyze", methods=["POST"])
def analyze_text():
    if "user" not in session:
        return jsonify({"error":"non loggato"}),403
    try:
        cv_text = request.form.get("cv_text","")
        job_req = request.form.get("job_requirements","")
        if not cv_text or not job_req:
            return jsonify({"error":"Mancano CV text e job requirements"}),400
        # ...
        return jsonify({"ok":"test"})
    except Exception as e:
        return jsonify({"error":str(e)}),500

########################################################################

if __name__=="__main__":
    app.run(debug=True)
