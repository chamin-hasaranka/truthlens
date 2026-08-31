"""
app.py
-------
Main Flask application for the Fake News Detection web app.

Features implemented in this file:
- User registration / login / logout with hashed passwords (Werkzeug)
- Session-based authentication
- SQLite database (auto-created on first run) for users + predictions
- Dashboard with prediction statistics
- Fake News prediction page (loads TF-IDF vectorizer + PassiveAggressiveClassifier via joblib)
- Prediction history page
- Admin panel (view users, view/delete predictions, view stats) - restricted to admin users
- Proper error handling throughout (try/except, flash messages, 404/500 handlers)

Run with:
    python app.py
"""

import os
import sqlite3
import joblib
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, g, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

# Import the same text-cleaning function used during training, to ensure
# the live prediction pipeline matches the training pipeline exactly.
from train_model import clean_text

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.pkl")

app = Flask(__name__)
# IMPORTANT: In production, set this via an environment variable instead of
# hardcoding it. Example: app.secret_key = os.environ.get("SECRET_KEY")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this-in-production-6f8a1c")

# ---------------------------------------------------------------------------
# Load ML model + vectorizer at startup (with graceful error handling)
# ---------------------------------------------------------------------------
model = None
vectorizer = None
model_metrics = {"accuracy": None}

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("[OK] Model and vectorizer loaded successfully.")
    else:
        print("[WARNING] Model/vectorizer not found. Run 'python train_model.py' first.")
except Exception as e:
    print(f"[ERROR] Failed to load model/vectorizer: {e}")

try:
    if os.path.exists(METRICS_PATH):
        model_metrics = joblib.load(METRICS_PATH)
except Exception as e:
    print(f"[WARNING] Failed to load metrics: {e}")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db():
    """Open a new database connection if one doesn't already exist for this request."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    """Close the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """
    Create database tables if they do not already exist, and seed a default
    admin account (admin@example.com / admin123) for convenience.
    """
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    db.commit()

    # Seed a default admin account if no admin exists yet
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
    admin_count = cursor.fetchone()[0]

    if admin_count == 0:
        cursor.execute(
            """INSERT INTO users (username, email, password_hash, is_admin, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "admin",
                "admin@example.com",
                generate_password_hash("admin123"),
                1,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()
        print("[OK] Default admin account created -> email: admin@example.com | password: admin123")

    db.close()


# ---------------------------------------------------------------------------
# Auth helpers / decorators
# ---------------------------------------------------------------------------
def login_required(view_func):
    """Decorator: redirect to login page if user is not authenticated."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def admin_required(view_func):
    """Decorator: only allow access if the logged-in user is an admin."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        if not session.get("is_admin"):
            flash("You do not have permission to access the admin panel.", "danger")
            return redirect(url_for("dashboard"))
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# ML prediction helper
# ---------------------------------------------------------------------------
def predict_news(text: str):
    """
    Run the loaded TF-IDF vectorizer + PassiveAggressiveClassifier on the
    given text and return (label, confidence_percentage).

    PassiveAggressiveClassifier does not natively output probabilities, so
    we use the decision_function (distance from the separating hyperplane)
    and squash it into a 0-100% confidence score using a sigmoid-like
    transformation. This gives a meaningful, monotonic confidence measure
    without requiring a probability-calibrated model.
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model is not loaded. Please run train_model.py first.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("The provided text is empty after cleaning. Please enter a valid news article.")

    features = vectorizer.transform([cleaned])
    raw_prediction = model.predict(features)[0]  # 0 = FAKE, 1 = REAL

    # decision_function returns the signed distance to the hyperplane
    decision_score = model.decision_function(features)[0]

    # Convert distance to a 0-1 "confidence" via a logistic squashing function
    import math
    confidence = 1 / (1 + math.exp(-abs(decision_score)))
    confidence_percentage = round(confidence * 100, 2)

    # Ensure confidence never looks artificially low for clearly separated points
    confidence_percentage = max(confidence_percentage, 50.01)

    label = "REAL" if raw_prediction == 1 else "FAKE"
    return label, confidence_percentage


# ---------------------------------------------------------------------------
# Routes: Public pages
# ---------------------------------------------------------------------------
@app.route("/")
def landing():
    """Landing / home page."""
    return render_template("landing.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration page."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # --- Basic server-side validation ---
        errors = []
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters long.")
        if not email or "@" not in email:
            errors.append("Please enter a valid email address.")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return render_template("register.html", username=username, email=email)

        try:
            db = get_db()
            cursor = db.cursor()

            # Check if username/email already exists
            cursor.execute(
                "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
            )
            if cursor.fetchone():
                flash("Username or email already exists. Please choose another.", "danger")
                return render_template("register.html", username=username, email=email)

            password_hash = generate_password_hash(password)
            cursor.execute(
                """INSERT INTO users (username, email, password_hash, is_admin, created_at)
                   VALUES (?, ?, ?, 0, ?)""",
                (username, email, password_hash, datetime.now().isoformat(timespec="seconds")),
            )
            db.commit()

            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))

        except sqlite3.Error as e:
            flash(f"Database error occurred: {e}", "danger")
            return render_template("register.html", username=username, email=email)
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")
            return render_template("register.html", username=username, email=email)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("login.html", email=email)

        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()

            if user is None or not check_password_hash(user["password_hash"], password):
                flash("Invalid email or password.", "danger")
                return render_template("login.html", email=email)

            # Set up session
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["is_admin"] = bool(user["is_admin"])

            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))

        except sqlite3.Error as e:
            flash(f"Database error occurred: {e}", "danger")
            return render_template("login.html", email=email)

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Log the user out by clearing the session."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("landing"))


# ---------------------------------------------------------------------------
# Routes: Authenticated pages
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard showing prediction statistics and recent history."""
    try:
        db = get_db()
        cursor = db.cursor()
        user_id = session["user_id"]

        cursor.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = ?", (user_id,))
        total_predictions = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM predictions WHERE user_id = ? AND prediction = 'REAL'",
            (user_id,),
        )
        real_count = cursor.fetchone()["cnt"]

        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM predictions WHERE user_id = ? AND prediction = 'FAKE'",
            (user_id,),
        )
        fake_count = cursor.fetchone()["cnt"]

        cursor.execute(
            """SELECT * FROM predictions WHERE user_id = ?
               ORDER BY created_at DESC LIMIT 5""",
            (user_id,),
        )
        recent_predictions = cursor.fetchall()

        model_accuracy = model_metrics.get("accuracy")
        model_accuracy_display = (
            f"{model_accuracy * 100:.2f}%" if model_accuracy is not None else "N/A"
        )

        return render_template(
            "dashboard.html",
            total_predictions=total_predictions,
            real_count=real_count,
            fake_count=fake_count,
            model_accuracy=model_accuracy_display,
            recent_predictions=recent_predictions,
        )

    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")
        return render_template(
            "dashboard.html",
            total_predictions=0,
            real_count=0,
            fake_count=0,
            model_accuracy="N/A",
            recent_predictions=[],
        )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """News prediction page: GET shows the form, POST runs the prediction."""
    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if not news_text:
            flash("Please enter or paste a news article to analyze.", "warning")
            return render_template("predict.html")

        if len(news_text) < 20:
            flash("Please enter a longer piece of text for a meaningful prediction (at least 20 characters).", "warning")
            return render_template("predict.html", news_text=news_text)

        try:
            label, confidence = predict_news(news_text)
            timestamp = datetime.now()

            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                """INSERT INTO predictions (user_id, news_text, prediction, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session["user_id"],
                    news_text,
                    label,
                    confidence,
                    timestamp.isoformat(timespec="seconds"),
                ),
            )
            db.commit()

            return render_template(
                "result.html",
                prediction=label,
                confidence=confidence,
                news_text=news_text,
                timestamp=timestamp.strftime("%B %d, %Y at %I:%M %p"),
            )

        except RuntimeError as e:
            flash(str(e), "danger")
            return render_template("predict.html", news_text=news_text)
        except ValueError as e:
            flash(str(e), "warning")
            return render_template("predict.html", news_text=news_text)
        except sqlite3.Error as e:
            flash(f"Database error occurred while saving prediction: {e}", "danger")
            return render_template("predict.html", news_text=news_text)
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")
            return render_template("predict.html", news_text=news_text)

    return render_template("predict.html")


@app.route("/history")
@login_required
def history():
    """Display the full prediction history for the logged-in user."""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """SELECT * FROM predictions WHERE user_id = ?
               ORDER BY created_at DESC""",
            (session["user_id"],),
        )
        predictions = cursor.fetchall()
        return render_template("history.html", predictions=predictions)

    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")
        return render_template("history.html", predictions=[])


@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
@login_required
def delete_history_item(prediction_id):
    """Allow a user to delete their own prediction history item."""
    try:
        db = get_db()
        cursor = db.cursor()
        # Ensure users can only delete their OWN predictions
        cursor.execute(
            "DELETE FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"]),
        )
        db.commit()

        if cursor.rowcount > 0:
            flash("Prediction deleted from your history.", "success")
        else:
            flash("Prediction not found or you do not have permission to delete it.", "warning")

    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")

    return redirect(url_for("history"))


# ---------------------------------------------------------------------------
# Routes: Admin panel
# ---------------------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_panel():
    """Admin dashboard: overall statistics, all users, all predictions."""
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("SELECT COUNT(*) AS cnt FROM users")
        total_users = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) AS cnt FROM predictions")
        total_predictions = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE prediction = 'REAL'")
        total_real = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE prediction = 'FAKE'")
        total_fake = cursor.fetchone()["cnt"]

        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()

        cursor.execute(
            """SELECT predictions.*, users.username FROM predictions
               JOIN users ON predictions.user_id = users.id
               ORDER BY predictions.created_at DESC"""
        )
        all_predictions = cursor.fetchall()

        model_accuracy = model_metrics.get("accuracy")
        model_accuracy_display = (
            f"{model_accuracy * 100:.2f}%" if model_accuracy is not None else "N/A"
        )

        return render_template(
            "admin.html",
            total_users=total_users,
            total_predictions=total_predictions,
            total_real=total_real,
            total_fake=total_fake,
            users=users,
            all_predictions=all_predictions,
            model_accuracy=model_accuracy_display,
        )

    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")
        return render_template(
            "admin.html",
            total_users=0,
            total_predictions=0,
            total_real=0,
            total_fake=0,
            users=[],
            all_predictions=[],
            model_accuracy="N/A",
        )


@app.route("/admin/delete_prediction/<int:prediction_id>", methods=["POST"])
@admin_required
def admin_delete_prediction(prediction_id):
    """Admin: delete any prediction by ID."""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        db.commit()
        flash("Prediction deleted successfully.", "success")
    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")

    return redirect(url_for("admin_panel"))


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    """Admin: delete a user (and their predictions) by ID. Cannot delete self."""
    try:
        if user_id == session["user_id"]:
            flash("You cannot delete your own admin account while logged in.", "warning")
            return redirect(url_for("admin_panel"))

        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        db.commit()
        flash("User and their prediction history deleted successfully.", "success")
    except sqlite3.Error as e:
        flash(f"Database error occurred: {e}", "danger")

    return redirect(url_for("admin_panel"))


# ---------------------------------------------------------------------------
# API endpoint (optional JSON prediction endpoint, useful for JS fetch calls)
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """JSON API endpoint for AJAX-based prediction (used by predict.html JS for live preview)."""
    try:
        data = request.get_json(silent=True) or {}
        news_text = data.get("news_text", "").strip()

        if not news_text or len(news_text) < 20:
            return jsonify({"error": "Please provide at least 20 characters of text."}), 400

        label, confidence = predict_news(news_text)
        return jsonify({"prediction": label, "confidence": confidence})

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5001)
