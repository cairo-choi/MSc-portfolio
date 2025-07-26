# 🧠 GPT-Enhanced Bitcoin Direction Prediction

This project implements a hybrid financial forecasting system that combines time series models (LSTM & GRU) with large language model (GPT-4) reasoning using Reddit sentiment headlines. The system predicts the **30-day directional movement** of Bitcoin (increase/decrease).

---

## 📌 Overview

We evaluate whether GPT-4 can act as a decision-making layer to:
- **Break ties** between conflicting LSTM and GRU predictions
- **Incorporate Reddit sentiment** as a tiebreaker or enhancement signal
- **Improve prediction metrics** such as accuracy, precision, recall, and F1 score

Both **train** and **test** phases are implemented with distinct GPT integration strategies to simulate realistic deployment scenarios.

---

## 📁 Project Structure

```
GPT_CODE/
│
├── data/
│   ├── reddit/
│   │   ├── train/                        # LSTM, GRU, Reddit and labels for training
│   │   └── test/                         # LSTM, GRU, Reddit and labels for testing
│   ├── reddit_daily_texts.csv            # Cleaned Reddit post headlines
│   ├── ohlcv_ma.csv                      # Bitcoin price features
│   ├── filtered_true_direction.csv       # Filtered ground truth labels
│   └── ...                               # Other support CSVs
│
├── predictions/
│   └── initial finding/                  # GPT-enhanced prediction outputs
│
├── run_GPT.py                            # Test set GPT-based decision generator
├── claude_train.py                       # Train set GPT fallback framework
├── evaluate_GPT_predictions.py           # Evaluation script (Accuracy, Precision, Recall, F1)
│
├── api.env                               # OpenAI API keys
├── praw.ini                              # Reddit API config (if using live scraping)
└── README.md                             # You're here!
```

---

## 🧪 Key Files

| File | Description |
|------|-------------|
| `claude_train.py` | Uses a **rule-based GPT prompt** to make decisions when LSTM and GRU disagree during training |
| `run_GPT.py` | Applies GPT uniformly across all test data points using Reddit text + model outputs |
| `evaluate_GPT_predictions.py` | Evaluates model/GPT decisions against ground truth labels using standard classification metrics |
| `data/reddit/` | Contains all model predictions and Reddit inputs for both training and testing |
| `predictions/` | Output CSVs containing GPT-enhanced predictions and metadata |

---

## 📊 Methodology Summary

- **Train phase**:  
  - If LSTM == GRU → take consensus  
  - If LSTM ≠ GRU and Reddit is missing → prefer GRU (higher F1)  
  - If LSTM ≠ GRU and Reddit is present → use GPT to reason

- **Test phase**:  
  - GPT always makes the final decision using all inputs

- **Reddit Data**:  
  - Cleaned Reddit headlines used as proxy for sentiment  
  - "No data" days are handled explicitly with placeholders

---

## 📈 Example Evaluation Output

```
📊 Evaluation Results
Accuracy: 0.76
Precision: 0.78
Recall: 0.72
F1 score: 0.75
```

---

## 🚀 Getting Started

### 1. Install dependencies  
Run:
```bash
pip install -r requirement files/requirements.txt
```

### 2. Set API Keys
Add your OpenAI API key to `api.env`:
```ini
OPENAI_API_KEY=your-key-here
```

### 3. Run predictions

For training:
```bash
python claude_train.py
```

For testing:
```bash
python run_GPT.py
```

### 4. Evaluate performance
```bash
python evaluate_GPT_predictions.py
```

---

## 📚 Citation & Academic Use

If you use this project in academic work, please cite it appropriately or acknowledge the authors in your publication.

---

## ⚠️ Limitations

- Sentiment is inferred indirectly (no labeled sentiment ground truth)
- GPT's reasoning is prompt-driven; results may vary slightly depending on temperature
- Focus is on interpretability and hybrid logic, not exhaustive model benchmarks

---

## 💡 Future Work

- Add transformer-based base models for comparison
- Explore fine-tuned sentiment classification to validate Reddit signals
- Test GPT-4's ability to generate rationales for each decision

---

## 📬 Contact

For questions or contributions, contact the author or raise an issue.
