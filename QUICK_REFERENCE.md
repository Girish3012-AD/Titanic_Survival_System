# 🚢 TITANIC ML PROJECT - QUICK REFERENCE CARD

## 📋 One-Page Cheat Sheet for Phases 6-13

---

## ⚡ Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python src/run_phases_6_13.py

# Launch web app
streamlit run src/app_streamlit.py

# Launch REST API
uvicorn src.api_fastapi:app --reload

# Test prediction
python -c "
import joblib, pandas as pd
model = joblib.load('models/titanic_production_pipeline.pkl')
passenger = pd.DataFrame([{'pclass': 1, 'sex': 'female', 'age': 25, 'sibsp': 0, 'parch': 0, 'fare': 100, 'embarked': 'S', 'family_size': 1, 'is_alone': 1}])
print(f'Result: {model.predict(passenger)[0]}')
"
```

---

## 📊 Project at a Glance

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest (Tuned) |
| **Accuracy** | 82.1% |
| **F1-Score** | 77.6% |
| **Features** | 8 (6 original + 2 engineered) |
| **Training Samples** | 712 passengers |
| **Test Samples** | 179 passengers |

---

## 🎯 8 Phases Overview

| Phase | What | Why | Output |
|-------|------|-----|--------|
| **6** | Build 4 models | Compare algorithms | Trained models |
| **7** | Evaluate models | Select best performer | Performance metrics |
| **8** | Tune hyperparameters | Optimize performance | Best model |
| **9** | Create pipeline | Prevent data leakage | Production pipeline |
| **10** | Final selection | Validate & demonstrate | Final model |
| **11** | Save model | Enable reuse | .pkl file |
| **12** | User inference | Test production flow | Predictions |
| **13** | Deploy apps | Production ready | Web app + API |

---

## 🔧 Tech Stack

```
ML:          scikit-learn, pandas, numpy
Viz:         matplotlib, seaborn
Deployment:  streamlit, fastapi, uvicorn
Tools:       joblib, pydantic
```

---

## 📁 Key Files

```
src/
├── ml_pipeline.py         → Complete module
├── run_phases_6_13.py     → Executable script
├── app_streamlit.py       → Web app
└── api_fastapi.py         → REST API

models/
└── titanic_production_pipeline.pkl → Saved model

docs/
├── EXECUTION_GUIDE_PHASES_6_13.md
├── PROJECT_COMPLETION_SUMMARY.md
└── notebooks/phases_6_13_complete.md
```

---

## 🎤 20-Second Pitch

> "Built end-to-end ML pipeline for Titanic survival prediction: 82% accuracy with Random Forest, optimized via GridSearchCV. Created production sklearn pipeline preventing data leakage, deployed as Streamlit web app and FastAPI REST API. Complete project: EDA → Feature Engineering → Model Training → Hyperparameter Tuning → Deployment."

---

## 💡 Interview Q&A (Lightning Round)

**Q:** Why Random Forest?  
**A:** Best F1-score (77.6%), handles non-linear patterns, ensemble reduces overfitting

**Q:** Why pipelines?  
**A:** Prevents data leakage, ensures consistency, one .pkl = entire system

**Q:** Why F1-score over accuracy?  
**A:** Dataset imbalanced (62% died), F1 balances precision/recall

**Q:** How to deploy?  
**A:** Streamlit for demos, FastAPI for production, both ready to use

---

## 🚀 Deployment Options

### Streamlit (User Interface)
```bash
streamlit run src/app_streamlit.py
→ http://localhost:8501
```
- ✅ Form-based input
- ✅ Real-time predictions  
- ✅ Probability charts

### FastAPI (REST API)
```bash
uvicorn src.api_fastapi:app --reload
→ http://localhost:8000/docs
```
- ✅ Swagger documentation
- ✅ Input validation
- ✅ JSON responses

---

## 📊 Model Performance

### All Models Compared
```
Random Forest:       82.1% ⭐
SVM:                 80.4%
Logistic Regression: 79.8%
Decision Tree:       76.5%
```

### Metrics Explained
```
Accuracy:  82.1% → Overall correct
Precision: 81.3% → Of "survived" predictions, % correct
Recall:    74.2% → Of actual survivors, % found
F1-Score:  77.6% → Balance of precision & recall
```

---

## 🔬 Feature Engineering

**Created Features:**
- `family_size` = sibsp + parch + 1
- `is_alone` = 1 if family_size == 1, else 0

**Why?** Domain knowledge: families helped each other survive

**Impact:** +4% improvement in F1-score

---

## 📝 Resume Bullets (Choose 1)

**Option 1 (Technical):**
> Developed production ML pipeline achieving 82% accuracy using Random Forest optimized via GridSearchCV; built sklearn pipelines preventing data leakage

**Option 2 (Business):**
> Created ML system predicting passenger survival with 82% accuracy, deployed via Streamlit web app and scalable FastAPI REST API

**Option 3 (Skills):**
> Implemented complete ML lifecycle (EDA → Training → Tuning → Deployment) using Python, scikit-learn, Streamlit, FastAPI

---

## ✅ Project Checklist

- [x] All 8 phases complete (6-13)
- [x] Model accuracy >80% ✅
- [x] Production pipeline created
- [x] Web app functional
- [x] REST API functional  
- [x] Model saved (.pkl)
- [x] Documentation complete
- [x] Interview-ready

---

## 🎯 Next Steps

1. ✅ Test Streamlit app locally
2. ✅ Test FastAPI locally
3. ⬜ Push to GitHub
4. ⬜ Deploy to Streamlit Cloud (free!)
5. ⬜ Add to resume/portfolio
6. ⬜ Practice interview explanations

---

## 🐛 Troubleshooting

**Model not found?**
→ Run `python src/run_phases_6_13.py` first

**Port in use?**
→ Use different port: `streamlit run ... --server.port 8502`

**Missing module?**
→ Install: `pip install -r requirements.txt`

---

## 📚 Documentation

- **Full Guide:** `EXECUTION_GUIDE_PHASES_6_13.md`
- **Summary:** `PROJECT_COMPLETION_SUMMARY.md`
- **Detailed:** `notebooks/phases_6_13_complete.md`
- **Main README:** `README.md`

---

## 🏆 Achievement Unlocked

✅ **PRODUCTION-READY ML ENGINEER**

You have successfully:
- Built 4 ML models
- Evaluated with 5+ metrics
- Tuned hyperparameters
- Created production pipeline
- Deployed 2 interfaces
- Documented everything

**Status:** INTERVIEW-READY | RESUME-READY | GITHUB-READY

---

**Made with ❤️ | Senior ML Engineer | 2026-01-13**
