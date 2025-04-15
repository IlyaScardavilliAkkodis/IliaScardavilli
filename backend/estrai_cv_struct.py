import pdfplumber
import re
from typing import Dict
from datetime import datetime

def correggi_errori_testo(testo: str) -> str:
    correzioni = {
        r'Handleba\s+rs': 'Handlebars',
        r'phpMyA\s+dmin': 'phpMyAdmin',
        r'Micro\s+soft': 'Microsoft',
        r'html\s+5': 'HTML5',
        r'css\s*3': 'CSS3',
        r'(?<=\w)\s+(?=\w)': '',  # unisci parole spezzate
    }
    for pattern, replacement in correzioni.items():
        testo = re.sub(pattern, replacement, testo, flags=re.IGNORECASE)
    return testo

def calcola_eta(data_str: str) -> str | None:
    try:
        data_nascita = datetime.strptime(data_str, "%d/%m/%Y")
        oggi = datetime.today()
        anni = oggi.year - data_nascita.year - ((oggi.month, oggi.day) < (data_nascita.month, data_nascita.day))
        return f"{anni} anni"
    except:
        return None

def estrai_testo_da_pdf(pdf_path: str) -> str:
    testo = ""
    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            estratto = pagina.extract_text(x_tolerance=2)
            if estratto:
                testo += estratto + "\n"
    return correggi_errori_testo(testo)

def estrai_info(testo: str) -> Dict:
    info = {
        'nome': None,
        'email': None,
        'telefono': None,
        'data_nascita': None,
        'eta': None,
        'nazionalita': None,
        'sesso': None,
        'indirizzo': None,
        'linkedin': None,
        'competenze': [],
        'esperienze': [],
    }

    nascita = re.search(r'data di nascita[:\s]*([0-3]?\d[\/\.-][01]?\d[\/\.-]\d{4})', testo, re.IGNORECASE)
    if nascita:
        data = nascita.group(1).replace('.', '/').replace('-', '/')
        info['data_nascita'] = data
        info['eta'] = calcola_eta(data)

    nazionalita = re.search(r'nazionalità[:\s]*(\w+)', testo, re.IGNORECASE)
    if nazionalita:
        info['nazionalita'] = nazionalita.group(1).capitalize()

    sesso = re.search(r'sesso[:\s]*(maschile|femminile)', testo, re.IGNORECASE)
    if sesso:
        info['sesso'] = sesso.group(1).capitalize()

    indirizzo = re.search(r'via\s+[A-Za-z\s]+\d+\s*\n?\d{5}\s+[A-Za-z\s]+', testo, re.IGNORECASE)
    if indirizzo:
        info['indirizzo'] = indirizzo.group(0).replace('\n', ', ').strip()

    linkedin = re.search(r'https:\/\/www\.linkedin\.com\/in\/[^\s]+', testo)
    if linkedin:
        info['linkedin'] = linkedin.group(0)

    righe = testo.splitlines()
    for riga in righe:
        if riga.strip() and riga.strip().istitle() and len(riga.strip().split()) >= 2:
            info['nome'] = riga.strip()
            break

    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', testo)
    if email:
        info['email'] = email.group(0)

    tel = re.search(r'(\+39\s?)?3\d{2}[\s\-]?\d{3}[\s\-]?\d{4}', testo)
    if tel:
        info['telefono'] = tel.group(0)

    if 'COMPETENZE' in testo.upper():
        sez = re.split(r'COMPETENZE.*', testo, flags=re.IGNORECASE)[-1]
        skills = re.findall(r'([A-Za-z0-9\+#\.]{3,})', sez)
        info['competenze'] = list(set(skills))

    esclusi = {'testing', 'competenze', 'sesso', 'esperienza', 'python', 'piani', 'universit', 'patente', 'altre'}
    esperienze = re.findall(r'(?:presso|in|dal|da)\s+([A-Z][a-zA-Z0-9\s,]+)', testo)
    uniche = set([e.strip() for e in esperienze if len(e.strip()) > 3])
    info['esperienze'] = [e for e in uniche if e.lower() not in esclusi]

    return info

def analizza_cv(pdf_path: str) -> Dict:
    testo = estrai_testo_da_pdf(pdf_path)
    info = estrai_info(testo)
    if not isinstance(info, dict):
        raise ValueError("Il risultato di analizza_cv non è un dizionario.")
    return info
