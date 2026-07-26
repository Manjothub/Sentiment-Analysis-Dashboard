# Sentiment Analysis Dashboard

A production-ready AI-powered Sentiment Analysis Dashboard that analyzes product reviews, classifies sentiment using DistilBERT, stores predictions in PostgreSQL, and displays analytics in a React dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [ML Model Setup & Training](#ml-model-setup--training)
6. [Running the Application](#running-the-application)
7. [Frontend Setup](#frontend-setup)
8. [API Documentation](#api-documentation)
9. [Database Setup](#database-setup)
10. [Testing](#testing)
11. [Project Structure](#project-structure)
12. [Troubleshooting](#troubleshooting)
13. [Configuration](#configuration)

---

## Project Overview

```
sentiment-dashboard/
│
├── backend/                    # Flask REST API + ML pipeline
│   ├── app/
│   │   ├── api/                # API route blueprints
│   │   ├── config/             # Environment-specific configuration
│   │   ├── database/           # Database connection & schema
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── services/           # Business logic services
│   │   └── utils/              # Logging & helpers
│   ├── scripts/                # Data preprocessing & EDA scripts
│   ├── tests/                  # Unit & integration tests
│   ├── logs/                   # Application logs
│   ├── requirements.txt        # Python runtime dependencies
│   ├── requirements-dev.txt    # Python development dependencies
│   ├── .env.example            # Environment template
│   ├── .env                    # Environment variables (create from .env.example)
│   └── app.py                  # Backend entry point
│
├── dataset/                    # Raw & processed datasets
│   ├── amazon-reviews/         # Amazon product reviews
│   ├── twitter-sentiment/      # Twitter sentiment data
│   └── processed/              # Cleaned/preprocessed data
│
├── ml_models/                  # Machine learning artifacts
│   ├── checkpoints/            # Training checkpoints
│   ├── saved_models/           # Trained model files (empty until trained)
│   └── training/               # Training configurations & logs
│
├── frontend/                   # React.js dashboard
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── services/           # API client & Socket.IO
│   │   ├── App.jsx             # Root component
│   │   └── main.jsx            # Entry point
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
└── README.md                   # This file
```

---

## Prerequisites

Install the following before starting:

| Software | Version | Download |
|----------|---------|----------|
| Python | 3.13 | [python.org/downloads](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| PostgreSQL | 15+ | [postgresql.org/download](https://www.postgresql.org/download/) |
| Git | Latest | [git-scm.com](https://git-scm.com/downloads) |

**Windows notes:**
- During Python installation, check **"Add Python to PATH"**.
- If using PowerShell, you may need to allow script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## Quick Start

### 1. Clone the Repository

```cmd
git clone <repository-url>
cd "Sentiment Analysis for Product Reviews"
```

### 2. Backend Setup

```cmd
cd backend

:: Create virtual environment with Python 3.13
py -3.13 -m venv venv

:: Activate virtual environment
venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install runtime dependencies
pip install -r requirements.txt

:: Install development dependencies (pytest, etc.)
pip install -r requirements-dev.txt

:: Download NLTK data
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('vader_lexicon', quiet=True)"

:: Create required directories
if not exist ..\ml_models\saved_models mkdir ..\ml_models\saved_models
if not exist ..\ml_models\checkpoints mkdir ..\ml_models\checkpoints
if not exist ..\ml_models\training mkdir ..\ml_models\training
if not exist ..\dataset\processed mkdir ..\dataset\processed
if not exist logs mkdir logs

cd ..
```

### 3. Configure Environment Variables

Copy the example environment file and edit it with your credentials:

```cmd
cd backend
copy .env.example .env
notepad .env
```

Update these values in `.env`:

```env
FLASK_ENV=development
SECRET_KEY=your-secure-random-key-here

DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentiment_dashboard
DB_USER=postgres
DB_PASSWORD=your-password-here

MODEL_DIR=../ml_models/saved_models
DATASET_DIR=../dataset
PROCESSED_DIR=../dataset/processed
```

### 4. Database Setup

```cmd
:: Log into PostgreSQL (use your password)
psql -U postgres

:: Create the database
CREATE DATABASE sentiment_dashboard;

:: Exit psql
\q
```

### 5. Train the ML Model (or Skip to Use Fallback)

The `ml_models/saved_models/` folder is empty. You must train the model before using `/api/predict`.

```cmd
cd backend

:: Step 1: Prepare dataset
python -c "from app.services.ml_pipeline.prepare_dataset import prepare_dataset; prepare_dataset()"

:: Step 2: Train DistilBERT
python -c "from app.services.ml_pipeline.train_model import train_distilbert; train_distilbert()"

:: Step 3: Evaluate model (optional)
python -c "from app.services.ml_pipeline.evaluate_model import evaluate_model; evaluate_model()"
```

Output:
- Model saved to `../ml_models/saved_models/`
- Tokenizer saved to `../ml_models/saved_models/tokenizer/`
- Checkpoints in `../ml_models/checkpoints/`
- Logs in `backend/logs/`

### 6. Frontend Setup

Open a **new terminal** (keep the backend venv active for later):

```cmd
cd frontend

:: Install Node.js dependencies
npm install

:: Build for production
npm run build
```

### 7. Run the Application

**Terminal 1 — Backend:**
```cmd
cd backend
venv\Scripts\activate
python app.py
```

**Terminal 2 — Frontend (development):**
```cmd
cd frontend
npm run dev
```

Open your browser:
- Frontend dashboard: `http://localhost:5173`
- Backend API: `http://localhost:5000`

---

## Detailed Setup

### Virtual Environment

**Windows CMD:**
```cmd
py -3.13 -m venv venv
venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3.13 -m venv venv
source venv/bin/activate
```

### Install Backend Dependencies

```cmd
cd backend
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For development tools (pytest, linters, Jupyter):
```cmd
pip install -r requirements-dev.txt
```

### Install Frontend Dependencies

```cmd
cd frontend
npm install
```

### Download NLTK Data

```cmd
cd backend
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('vader_lexicon', quiet=True)"
```

### Create Required Directories

```cmd
cd backend
if not exist ..\ml_models\saved_models mkdir ..\ml_models\saved_models
if not exist ..\ml_models\checkpoints mkdir ..\ml_models\checkpoints
if not exist ..\ml_models\training mkdir ..\ml_models\training
if not exist ..\dataset\processed mkdir ..\dataset\processed
if not exist logs mkdir logs
```

---

## ML Model Setup & Training

The `ml_models/saved_models/` directory is empty by default. The application runs in **fallback mode** (rule-based sentiment) until a trained model is present.

### Option A: Train from Scratch

**Step 1: Prepare Dataset**
```cmd
cd backend
python -c "from app.services.ml_pipeline.prepare_dataset import prepare_dataset; prepare_dataset()"
```

This will:
- Load `dataset/processed/amazon_cleaned.csv`
- Perform stratified 80/10/10 train/val/test split
- Tokenize with DistilBERT tokenizer (max_length=256)
- Save to `dataset/processed/hf_dataset/`
- Save tokenizer to `ml_models/saved_models/tokenizer/`

**Step 2: Fine-Tune DistilBERT**
```cmd
python -c "from app.services.ml_pipeline.train_model import train_distilbert; train_distilbert()"
```

Training defaults:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 10 (with early stopping, patience=3) |
| Max Length | 256 tokens |
| Mixed Precision | FP16 (if GPU available) |
| Gradient Accumulation | 2 steps |

Output:
- Best model → `ml_models/saved_models/`
- Tokenizer → `ml_models/saved_models/tokenizer/`
- Checkpoints → `ml_models/checkpoints/`
- TensorBoard logs → `backend/logs/tensorboard/`
- Model registered in database `model_versions` table

**Step 3: Evaluate Model (Optional)**
```cmd
python -c "from app.services.ml_pipeline.evaluate_model import evaluate_model; evaluate_model()"
```

Generates:
- Accuracy, Precision, Recall, F1
- Confusion Matrix PNG
- ROC Curves PNG
- Classification Report JSON
- All saved to `backend/logs/evaluation/`

### Option B: Use Rule-Based Fallback

If you don't want to train the model, the backend automatically falls back to keyword-based sentiment analysis. No model files are required, but predictions will be less accurate.

---

## Running the Application

### Activate Backend Virtual Environment

```cmd
cd backend
venv\Scripts\activate
```

### Start Backend Server

```cmd
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

**Production serving (Windows):**
```cmd
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 backend.app:app
```

**Production serving (Linux/macOS):**
```cmd
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

### Start Frontend Dev Server

Open a new terminal:

```cmd
cd frontend
npm run dev
```

The dashboard will be available at `http://localhost:5173`

### Build Frontend for Production

```cmd
cd frontend
npm run build
```

The built files will be in `frontend/dist/`.

---

## Frontend Setup

### Install Dependencies

```cmd
cd frontend
npm install
```

### Development Server

```cmd
npm run dev
```

### Production Build

```cmd
npm run build
npm run preview
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server (HMR enabled) |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build locally |

---

## API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Application health check |
| `/api/model/status` | GET | Model loading status & device info |
| `/api/model/info` | GET | Model metadata & training history |
| `/api/predict` | POST | Single review sentiment prediction |
| `/api/batch_predict` | POST | Batch sentiment prediction (max 100) |
| `/api/topics` | GET | Topic modeling results |
| `/api/aspects` | GET | Aggregate aspect analysis |
| `/api/stats` | GET | Dashboard statistics |
| `/api/reviews` | GET | Paginated reviews with sentiment |
| `/api/trending-topics` | GET | Trending keywords with sentiment |
| `/api/trends-over-time` | GET | Sentiment trends over time |
| `/api/alerts` | GET | Alert list |
| `/api/alerts/summary` | GET | Alert summary statistics |
| `/api/dashboard/overview` | GET | Combined dashboard data |

### Example: POST /api/predict

**Request:**
```json
{
    "text": "This product is amazing! Great quality and fast shipping.",
    "include_aspects": true,
    "store_result": false,
    "product_id": "B08N5WRWNW"
}
```

**Response:**
```json
{
    "success": true,
    "text": "This product is amazing! Great quality and fast shipping.",
    "sentiment": {
        "predicted_sentiment": "positive",
        "positive_score": 0.95,
        "negative_score": 0.02,
        "neutral_score": 0.03,
        "confidence_score": 0.95,
        "inference_time_ms": 45
    },
    "aspects": {
        "product_quality": "positive",
        "shipping": "positive"
    },
    "metadata": {
        "total_inference_time_ms": 245,
        "model_loaded": true,
        "timestamp": "2024-01-15T10:30:00"
    }
}
```

---

## Database Setup

### Create Database

```cmd
psql -U postgres -c "CREATE DATABASE sentiment_dashboard;"
```

### Environment Variables

Edit `backend/.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentiment_dashboard
DB_USER=postgres
DB_PASSWORD=your-password
```

### Database Migrations

```cmd
cd backend
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

The app also auto-creates tables on startup via `db.create_all()`.

### Schema Overview

| Table | Description |
|-------|-------------|
| `products` | Product catalog |
| `reviews` | Raw review text & metadata |
| `sentiments` | Sentiment predictions & aspects |
| `model_versions` | Trained model registry |

---

## Testing

### Backend Tests

```cmd
cd backend
venv\Scripts\activate
pytest tests/test_ml_pipeline.py -v      # 12 tests
pytest tests/test_sentiment_service.py -v # 25 tests
pytest tests/test_api.py -v              # 18 tests
pytest tests/ -v                         # All tests
```

### Frontend Build Check

```cmd
cd frontend
npm run build
```

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── config/
│   │   └── settings.py          # Environment configs
│   ├── database/
│   │   ├── __init__.py          # SQLAlchemy & Flask-Migrate init
│   │   └── schema.sql           # Raw SQL schema
│   ├── models/
│   │   ├── __init__.py          # db + socketio instances
│   │   ├── product.py           # Product ORM
│   │   ├── review.py            # Review ORM
│   │   ├── sentiment.py         # SentimentResult ORM
│   │   └── model_version.py     # ModelVersion ORM
│   ├── services/
│   │   ├── nlp_service.py       # Legacy NLP service
│   │   ├── alert_service.py     # Alert logic
│   │   └── ml_pipeline/
│   │       ├── prepare_dataset.py
│   │       ├── train_model.py
│   │       ├── evaluate_model.py
│   │       ├── sentiment_service.py
│   │       ├── aspect_service.py
│   │       └── topic_model.py
│   ├── utils/
│   │   └── logger.py            # Logging setup
│   └── api/
│       ├── __init__.py
│       ├── health.py
│       ├── model.py
│       ├── sentiment.py
│       ├── trends.py
│       ├── alerts.py
│       └── comparative.py
├── scripts/
│   ├── run_preprocessing.py
│   ├── preprocess_amazon.py
│   ├── preprocess_twitter.py
│   ├── exploratory_analysis.py
│   └── inspect_datasets.py
├── tests/
│   ├── test_api.py
│   ├── test_ml_pipeline.py
│   └── test_sentiment_service.py
├── app.py                       # Entry point
├── requirements.txt
├── requirements-dev.txt
└── .env.example

frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx
│   │   ├── SentimentChart.jsx
│   │   ├── SentimentGauge.jsx
│   │   ├── AspectAnalysis.jsx
│   │   ├── TrendingTopics.jsx
│   │   ├── AlertPanel.jsx
│   │   └── ComparativeAnalysis.jsx
│   ├── services/
│   │   └── api.js               # Axios + Socket.IO client
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── package.json
├── vite.config.js
└── index.html
```

---

## Troubleshooting

### 1. Virtual Environment Activation Fails

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows CMD:**
```cmd
:: Use CMD instead of PowerShell, or run:
venv\Scripts\activate.bat
```

### 2. PostgreSQL Connection Error

```cmd
:: Start PostgreSQL service
net start postgresql-x64-15

:: Test connection
psql -U postgres -d sentiment_dashboard

:: Verify credentials in backend/.env
```

### 3. Module Not Found Errors

```cmd
cd backend
venv\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

### 4. Dataset Not Found

```cmd
:: Ensure preprocessing was run first
cd backend
python scripts/run_preprocessing.py --amazon-nrows 10000
```

### 5. Memory Errors During Training

- Reduce batch size: `TrainingConfig(batch_size=8)`
- Enable gradient checkpointing
- Use CPU fallback: `TrainingConfig(fp16=False)`

### 6. CUDA Out of Memory

- Reduce `batch_size` and `eval_batch_size`
- Reduce `max_length` to 128
- Enable gradient checkpointing
- Use CPU instead

### 7. pip Install Slow / Fails

Use a mirror or install packages individually:

```cmd
:: Use default PyPI
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-cache-dir
```

### 8. npm Install Fails on Windows

```cmd
:: Clear npm cache
npm cache clean --force

:: Delete node_modules and reinstall
rd /s /q node_modules
del package-lock.json
npm install
```

---

## Configuration

### Environment Variables

Create `backend/.env` from `backend/.env.example`:

```env
FLASK_ENV=development
SECRET_KEY=your-secure-random-key-here

DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentiment_dashboard
DB_USER=postgres
DB_PASSWORD=your-password-here

MODEL_DIR=../ml_models/saved_models
DATASET_DIR=../dataset
PROCESSED_DIR=../dataset/processed
```

### Training Configuration

Override defaults by creating a custom `TrainingConfig`:

```python
from app.services.ml_pipeline.train_model import TrainingConfig, train_distilbert

config = TrainingConfig(
    batch_size=8,
    num_epochs=5,
    learning_rate=1e-5,
    fp16=False,
    output_dir='../ml_models/saved_models/custom'
)
train_distilbert(config)
```

---

## License

MIT License
# Sentiment-Analysis-Dashboard
