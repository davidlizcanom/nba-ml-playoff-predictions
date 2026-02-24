# nba-ml-playoff-predictions
# 🏀 NBA Playoffs Predictions with Machine Learning & Monte Carlo Simulation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-1.7+-orange?logo=xgboost" />
  <img src="https://img.shields.io/badge/Monte%20Carlo-10%2C000%20sims-green" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  <strong>An end-to-end machine learning pipeline that predicts NBA Playoff outcomes by training an XGBoost model on 10 seasons of historical data and running 10,000 Monte Carlo simulations of the full playoff bracket.</strong>
</p>

<p align="center">
  <a href="#-key-results">Key Results</a> •
  <a href="#-methodology">Methodology</a> •
  <a href="#-notebooks">Notebooks</a> •
  <a href="#-how-to-run">How to Run</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

---

## 📊 Key Results (2025-26 Season)

| Rank | Team | Championship % | Finals % | Conf Finals % |
|------|------|---------------|----------|---------------|
| 1 | Oklahoma City Thunder | **60.5%** | 63.3% | 88.4% |
| 2 | Detroit Pistons | **17.1%** | 66.8% | 85.2% |
| 3 | San Antonio Spurs | **13.2%** | 25.8% | 67.8% |
| 4 | Denver Nuggets | 3.0% | 9.0% | 31.5% |
| 5 | New York Knicks | 2.6% | 14.0% | 56.0% |

> **Most likely NBA Finals:** Detroit Pistons vs Oklahoma City Thunder (42.2% of simulations)

### Key Findings

- **Dominant favorite:** OKC at 60.5% — a level of dominance rarely seen in modern NBA playoff predictions.
- **Detroit paradox:** The Pistons reach the Finals *more often* than OKC (66.8% vs 63.3%) but win the championship far less — they dominate the East but can't overcome OKC.
- **Most predictive feature:** Effective Field Goal Percentage (eFG%) — not points scored, not defensive rating. In playoffs, shooting efficiency is king.
- **First round upset to watch:** Los Angeles Lakers over Houston Rockets (43.1%) — the Lakers have the worst Net Rating of all playoff teams but the highest clutch factor in the league (90%).

<p align="center">
  <img src="outputs/championship_distribution.png" width="700" alt="Championship probability distribution" />
</p>

---

## 🔬 Methodology

### Pipeline Overview

```
Data Collection → Feature Engineering → Model Training → Validation → Monte Carlo Simulation
     (NB01)           (NB02)              (NB03)          (NB03)          (NB04)
```

### 1. Data Collection
- **Source:** Official NBA statistics via [`nba_api`](https://github.com/swar/nba_api)
- **Current season:** Team stats, game logs, standings (2025-26)
- **Historical:** 10 seasons of playoff results, team stats, and standings (2015-16 to 2024-25)

### 2. Feature Engineering
Features are computed as **differentials** (Team A − Team B), where Team A is always the higher seed:

| Feature | Description |
|---------|-------------|
| `EFG_PCT_diff` | Effective Field Goal % differential |
| `W_PCT_diff` | Win percentage differential |
| `TS_PCT_diff` | True Shooting % differential |
| `PIE_diff` | Player Impact Estimate differential |
| `seed_diff` | Playoff seed differential |

**Design decisions:**
- Differentials (not absolutes) → model learns "better *than opponent*"
- Team A = better seed → standardizes perspective
- Inverted sign for defensive stats → positive always means "Team A is better"
- Feature selection reduced from 14 to 5 → with ~150 training samples, fewer features = better generalization

### 3. Model: XGBoost

```python
XGBClassifier(
    n_estimators=50, max_depth=2,        # Shallow trees → less overfitting
    learning_rate=0.05, subsample=0.7,
    reg_alpha=2.0, reg_lambda=3.0,       # Strong L1 + L2 regularization
    min_child_weight=5, gamma=0.5        # Conservative splitting
)
```

**Aggressively regularized** for a small dataset (~150 series). Priority: reliable probabilities over maximum accuracy.

### 4. Validation: Leave-One-Season-Out (LOSO)

Train on 9 seasons, predict the remaining one. Repeat for each season. This simulates the real scenario of predicting future playoffs.

| Metric | Value |
|--------|-------|
| **OOF Accuracy** | 0.707 (matches historical baseline) |
| **Brier Score** | 0.1845 (< 0.20 = well-calibrated) |
| **ROC AUC** | 0.714 |
| **Probability range** | 0.446 – 0.885 (good discrimination) |

> **Why Brier Score matters more than accuracy here:** Monte Carlo simulation needs *calibrated probabilities*, not just binary predictions. A model that says "65% for Team A" must mean Team A wins ~65% of the time in similar situations.

<p align="center">
  <img src="outputs/feature_importance.png" width="600" alt="Feature importance" />
</p>

### 5. Monte Carlo Simulation

- **10,000 full bracket simulations** (16 teams, 4 rounds per conference + Finals)
- Each series: **Best-of-7**, simulated game by game
- **Home court advantage:** +3% per game for the higher seed (2-2-1-1-1 format)
- **Probability clipping:** 5%–95% (no certainties in playoffs)
- Pre-cached matchup probabilities for efficiency

---

## 📓 Notebooks

The project is organized in 4 sequential notebooks designed to run in **Google Colab**:

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [`01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) | Collects current season stats and 10 years of historical playoff data via `nba_api`. Saves to Google Drive. |
| 02 | [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | Builds training dataset with differential features. EDA, correlation analysis, and current season team profiles. |
| 03 | [`03_model_calibration.ipynb`](notebooks/03_model_calibration.ipynb) | Feature selection, XGBoost training, LOSO validation, calibration analysis, and historical backtest. |
| 04 | [`04_simulation_and_viz.ipynb`](notebooks/04_simulation_and_viz.ipynb) | 10,000 Monte Carlo simulations, championship probabilities, bracket visualization, and insights. |

### Data Flow

```
NB01 → Google Drive (data/)
         ↓
NB02 → Google Drive (data/ + team_profiles)
         ↓
NB03 → Google Drive (models/ + metrics)
         ↓
NB04 → Google Drive (outputs/ + visualizations)
```

---

## 🚀 How to Run

### Prerequisites
- Google account (for Colab + Drive)
- No local installation needed

### Steps

1. **Clone or download** this repository
2. **Upload the 4 notebooks** to [Google Colab](https://colab.research.google.com/)
3. **Run in order:** NB01 → NB02 → NB03 → NB04
4. Each notebook mounts Google Drive and saves outputs to `MyDrive/nba-playoffs-simulator/`
5. After NB01, subsequent notebooks read data from Drive — no need to re-run NB01 unless updating source data

### Requirements

```
nba_api>=1.4
xgboost>=1.7
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
```

> All dependencies are installed automatically in the first cell of each notebook via `!pip install`.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Model** | XGBoost (gradient boosting) |
| **Simulation** | Monte Carlo (NumPy) |
| **Data Source** | nba_api (official NBA stats) |
| **Validation** | Leave-One-Season-Out CV (scikit-learn) |
| **Visualization** | Matplotlib, Seaborn |
| **Platform** | Google Colab |
| **Storage** | Google Drive |

---

## 📁 Repository Structure

```
nba-ml-playoff-predictions/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_calibration.ipynb
│   └── 04_simulation_and_viz.ipynb
├── outputs/
│   ├── championship_distribution.png
│   ├── feature_importance.png
│   ├── playoff_bracket.png
│   ├── finals_matchups.png
│   ├── calibration_plot.png
│   └── validation_by_season.png
├── docs/
│   └── methodology.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📈 Historical Validation (Backtest)

The model was backtested on the last 3 completed seasons. For each, the model was trained *excluding* that season and then ran 5,000 simulations:

| Season | Actual Champion | Model's Top 5? |
|--------|----------------|-----------------|
| 2022-23 | Denver Nuggets | ✅ |
| 2023-24 | Boston Celtics | ✅ |
| 2024-25 | — | ✅ |

---

## 💡 Limitations & Future Work

### Limitations
- **Regular season stats only** — the model doesn't capture "playoff DNA" (player experience, coaching adjustments). Roster changes between seasons make historical playoff performance unreliable as a feature.
- **Small training set** (~150 series) — limits model complexity. Feature selection was critical to avoid overfitting.
- **No player-level data** — injuries, trades, and lineup changes during the season aren't modeled.

### Future Work
- Incorporate player-level features (injury-adjusted metrics)
- Add Elo-style momentum features from recent games
- Bayesian calibration for better probability estimates
- Web app for interactive bracket exploration

---

## 👤 Author

**David** — Data Analytics Master's Student | [LinkedIn](https://linkedin.com/in/YOUR-PROFILE) | [TikTok](https://tiktok.com/@YOUR-HANDLE)

Built as part of a data science content strategy combining technical analysis with sports analytics.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>If you found this useful, give it a ⭐ and share your team's chances in the Issues tab!</i>
</p>
