# Titanic Survival Prediction — DVC ML Pipeline

A reproducible machine learning pipeline for predicting Titanic passenger survival, built with DVC for experiment tracking and pipeline management.

**Course:** DS-4491 — Machine Learning Systems Design

---

## Problem & Dataset

**Problem:** Binary classification — predict whether a passenger survived the Titanic disaster based on their attributes (class, sex, age, fare, etc.).

**Dataset:** [Titanic dataset](https://www.kaggle.com/c/titanic) — 891 passengers with 12 columns.

| Column | Description |
|---|---|
| PassengerId | Unique ID (dropped during split) |
| Survived | Target variable (0 = died, 1 = survived) |
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Name | Passenger name (dropped during preprocessing) |
| Sex | Gender (encoded to binary: female=1, male=0) |
| Age | Age in years (missing values imputed with median) |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number (dropped during preprocessing) |
| Fare | Passenger fare (missing values imputed with median) |
| Cabin | Cabin number (converted to binary HasCabin feature) |
| Embarked | Port of embarkation (one-hot encoded: C, Q, S) |

---

## ML Model

**Algorithm:** Random Forest Classifier (scikit-learn)

**Hyperparameters** (configured in `params.yaml`):

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 12 |
| `random_state` | 42 |

**Features used (12 numeric features after preprocessing):**

Pclass, Sex, Age, SibSp, Parch, Fare, FamilySize, IsAlone, HasCabin, Embarked_C, Embarked_Q, Embarked_S

---

## Project Structure

```
MLSD Project/
├── data/
│   ├── titanic.csv              # Raw dataset (DVC-tracked)
│   ├── titanic.csv.dvc          # DVC pointer file
│   ├── interim/                 # Raw train/test split
│   │   ├── train.csv
│   │   └── test.csv
│   └── prepared/                # Preprocessed train/test data
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── split.py                 # Stratified train/test split
│   ├── preprocess.py            # Feature engineering & imputation
│   ├── train.py                 # Random Forest training
│   └── evaluate.py              # Model evaluation & metrics
├── models/
│   └── model.joblib             # Trained model artifact
├── metrics/
│   └── metrics.json             # Evaluation metrics
├── .dvc/
│   └── config                   # DVC remote configuration (DagsHub)
├── dvc.yaml                     # Pipeline definition
├── dvc.lock                     # Pipeline lock file (hashes)
├── params.yaml                  # Hyperparameters
├── requirements.txt             # Python dependencies
└── README.md
```

---

## DVC Pipeline

The project uses a 4-stage DVC pipeline:

```
data/titanic.csv
       │
       ▼
  ┌─────────┐   params: split.test_size, split.random_state
  │  split   │   Stratified 80/20 split
  └─────────┘
       │
       ▼
 ┌────────────┐
 │ preprocess │   Imputation, encoding, feature engineering
 └────────────┘
       │
       ▼
  ┌─────────┐   params: train.n_estimators, train.max_depth, train.random_state
  │  train   │   Random Forest Classifier
  └─────────┘
       │
       ▼
 ┌──────────┐
 │ evaluate  │   Accuracy, Precision, Recall, F1
 └──────────┘
       │
       ▼
  metrics.json
```

**Stages:**

| Stage | Script | Description |
|---|---|---|
| `split` | `src/split.py` | Splits raw data into train (80%) and test (20%) with stratification on Survived |
| `preprocess` | `src/preprocess.py` | Encodes Sex, imputes Age/Fare with train-set median, one-hot encodes Embarked, engineers FamilySize/IsAlone/HasCabin |
| `train` | `src/train.py` | Trains a Random Forest classifier and saves the model |
| `evaluate` | `src/evaluate.py` | Evaluates the model on the test set and writes metrics to JSON |

---

## How to Run

### Prerequisites

- Python 3.10+
- Git
- DVC

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/<repo>.git
cd mlsd-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Pull data from DagsHub
dvc pull
```

### Reproduce the Pipeline

```bash
# Run the full pipeline
dvc repro

# Check pipeline status
dvc status

# View the pipeline DAG
dvc dag

# View metrics
dvc metrics show
```

### Useful DVC Commands

```bash
dvc status      # Check if any stage needs re-running
dvc repro       # Reproduce the pipeline (re-runs changed stages)
dvc dag         # Visualize the pipeline graph
dvc push        # Push data/artifacts to DagsHub remote
dvc pull        # Pull data/artifacts from DagsHub remote
dvc metrics show  # Display model evaluation metrics
```

---

## Results

| Metric | Value |
|---|---|
| **Accuracy** | 0.7765 |
| **Precision** | 0.7101 |
| **Recall** | 0.7101 |
| **F1 Score** | 0.7101 |

---

## Technologies Used

- **Python** — Programming language
- **DVC** — Data versioning & pipeline management
- **Git / GitHub** — Code versioning
- **DagsHub** — DVC remote storage
- **pandas** — Data manipulation
- **scikit-learn** — Machine learning (Random Forest)
- **joblib** — Model serialization
- **PyYAML** — Configuration parsing
