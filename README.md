# TruthLens AI — Fake News Detection Web Application

A complete, production-ready **AI-powered Fake News Detection System** built with
**Python, Flask, scikit-learn, SQLite, Bootstrap 5, HTML/CSS/JavaScript**.

Users can register an account, paste a news article, and instantly get an AI-generated
**REAL** or **FAKE** classification with a confidence score. All predictions are stored
in a personal history, displayed on a statistics dashboard, and a full admin panel
lets administrators manage users and predictions platform-wide.

---

## ✨ Features

- **Modern Landing Page** — animated hero section, gradient backgrounds, feature cards, about section, footer
- **User Authentication** — registration, login, hashed passwords (Werkzeug), session-based auth, logout
- **Dashboard** — total predictions, REAL/FAKE counts, model accuracy, recent history
- **AI Prediction Engine** — TF-IDF vectorizer + PassiveAggressiveClassifier, trained via `train_model.py`
- **Confidence Scoring** — every prediction includes a transparent confidence percentage
- **Prediction History** — full history per user, with delete capability
- **Admin Panel** — view all users & predictions, delete any record, platform-wide stats
- **SQLite Database** — auto-created on first run, stores users + predictions with timestamps
- **Modern UI** — Bootstrap 5, dark/light theme toggle, smooth scroll animations, fully responsive

---

## 📁 Project Structure

```
FakeNewsDetection/
│
├── app.py                      # Main Flask application (routes, auth, DB, ML inference)
├── train_model.py              # ML training script (TF-IDF + PassiveAggressiveClassifier)
├── requirements.txt            # Python dependencies
├── database.db                 # SQLite database (auto-created on first run)
│
├── dataset/
│   ├── generate_sample_dataset.py   # Generates a sample Fake.csv/True.csv for quick start
│   ├── Fake.csv                     # Fake news training data
│   └── True.csv                     # Real news training data
│
├── model/
│   ├── fake_news_model.pkl     # Trained classifier (created by train_model.py)
│   ├── tfidf_vectorizer.pkl    # Trained TF-IDF vectorizer (created by train_model.py)
│   └── metrics.pkl             # Saved accuracy/precision/recall metrics
│
├── static/
│   ├── css/style.css           # Full design system (gradients, dark mode, animations)
│   ├── js/script.js            # Theme toggle, scroll animations, password visibility
│   └── images/                 # Static image assets (optional, empty by default)
│
├── templates/
│   ├── base.html                # Shared layout (navbar, flash messages, footer)
│   ├── landing.html             # Landing / home page
│   ├── login.html                # Login page
│   ├── register.html             # Registration page
│   ├── dashboard.html            # User dashboard with stats
│   ├── predict.html              # Prediction input form
│   ├── result.html               # Prediction result page
│   ├── history.html              # Prediction history page
│   ├── admin.html                # Admin panel
│   ├── 404.html                  # Not found error page
│   └── 500.html                  # Server error page
│
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed
- `pip` package manager

### 2. Clone / Download the Project
Place the `FakeNewsDetection` folder anywhere on your machine, then open a terminal
inside it.

### 3. Create a Virtual Environment (recommended)
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Prepare the Dataset
A small sample dataset is already included so the app works out of the box. If you
want to regenerate it (or you deleted it), run:
```bash
python dataset/generate_sample_dataset.py
```

**For best real-world accuracy**, replace the sample CSVs with the full
[Kaggle "Fake and Real News" dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset):
download `Fake.csv` and `True.csv` and place them inside the `dataset/` folder
(overwriting the sample files). The column schema (`title`, `text`, `subject`, `date`)
is identical, so no code changes are required.

### 6. Train the Machine Learning Model
```bash
python train_model.py
```
This will:
- Load and clean the dataset
- Split into train/test sets
- Vectorize text with TF-IDF
- Train a PassiveAggressiveClassifier
- Print accuracy/precision/recall/F1 metrics
- Save `fake_news_model.pkl` and `tfidf_vectorizer.pkl` into the `model/` folder

### 7. Run the Application
```bash
python app.py
```
The app will automatically create `database.db` (with the required tables) on
first run, and seed a default **admin account**:

```
Email:    admin@example.com
Password: admin123
```

Then open your browser to:
```
http://127.0.0.1:5000
```

> ⚠️ **Important:** Change the default admin password and the Flask `secret_key`
> (in `app.py`, or via the `SECRET_KEY` environment variable) before deploying
> this application publicly.

---

## 🗄️ Database Schema (SQLite)

The `init_db()` function in `app.py` creates the following tables automatically:

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    news_text TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

---

## 🧠 How the Machine Learning Pipeline Works

1. **Data Loading** — `Fake.csv` (label 0) and `True.csv` (label 1) are loaded and merged.
2. **Text Cleaning** — lowercasing, URL/HTML/punctuation/number removal, whitespace normalization (`clean_text()` in `train_model.py`, reused identically in `app.py` for live predictions to guarantee consistent preprocessing).
3. **Train/Test Split** — 80/20 stratified split.
4. **TF-IDF Vectorization** — converts text into weighted term-frequency vectors (`stop_words="english"`, unigrams + bigrams, `max_features=50000`).
5. **Model Training** — a `PassiveAggressiveClassifier` (an efficient online-learning linear classifier well suited to high-dimensional sparse text data) is trained on the vectorized data.
6. **Evaluation** — accuracy, precision, recall, and F1 score are computed on the held-out test set and saved to `model/metrics.pkl` (displayed on the Dashboard and Admin panel).
7. **Persistence** — both the trained `model` and `vectorizer` are serialized with **Joblib** into the `model/` folder.
8. **Inference** — `app.py` loads both `.pkl` files at startup. For each prediction request, text is cleaned the same way, vectorized, and classified. Since `PassiveAggressiveClassifier` doesn't natively output probabilities, the model's `decision_function` (signed distance from the decision boundary) is passed through a logistic squashing function to produce an interpretable 0–100% confidence score.

---

## 🔐 Security Notes

- Passwords are hashed using Werkzeug's `generate_password_hash` / `check_password_hash` (PBKDF2) — plaintext passwords are never stored.
- Sessions are signed using Flask's built-in secure session cookie mechanism (`app.secret_key`).
- All database queries use parameterized statements to prevent SQL injection.
- Route-level decorators (`@login_required`, `@admin_required`) enforce access control on every protected page.
- Users can only delete their **own** prediction history; only admins can delete any user's data.

For production deployment, additionally consider:
- Serving behind HTTPS (e.g. via Nginx + Gunicorn/Waitress)
- Setting `SECRET_KEY` via an environment variable, not hardcoded
- Disabling Flask debug mode (`debug=False`)
- Adding rate-limiting on login/register endpoints
- Migrating from SQLite to PostgreSQL/MySQL for multi-user production scale

---

## 🚀 Run Commands Summary

| Action                         | Command                                   |
|--------------------------------|--------------------------------------------|
| Install dependencies           | `pip install -r requirements.txt`           |
| Generate sample dataset        | `python dataset/generate_sample_dataset.py` |
| Train the ML model             | `python train_model.py`                     |
| Run the web application        | `python app.py`                             |
| Access the app                 | `http://127.0.0.1:5000`                     |

---

## 🧪 Default Demo Credentials

| Role  | Email                | Password   |
|-------|-----------------------|------------|
| Admin | admin@example.com     | admin123   |

(Regular users can self-register via the **Register** page.)

---

## 📦 Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (TF-IDF, PassiveAggressiveClassifier), pandas, Joblib
- **Database:** SQLite
- **Frontend:** HTML5, CSS3 (custom design system), JavaScript (vanilla), Bootstrap 5, Bootstrap Icons

---

## 📝 License & Disclaimer

This project is provided for educational and portfolio purposes. The fake news
classifier is a statistical model trained on a limited dataset — predictions are
probabilistic estimates, not definitive fact-checking, and should not be relied
upon as the sole source of truth for verifying news. Always cross-reference
important information with trusted, primary sources.
