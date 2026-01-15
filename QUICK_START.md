# 🚀 Quick Start Guide - Titanic ML Project

## Option 1: Quick Setup (5 minutes)

### Step 1: Install Dependencies
```powershell
# Navigate to project directory
cd C:\Users\Lenovo\Desktop\mlpro

# Install required packages
pip install -r requirements.txt
```

### Step 2: Launch Jupyter Notebook
```powershell
jupyter notebook
```

### Step 3: Run the Project
- Open `notebooks/titanic_ml_project.ipynb`
- Click **Kernel → Restart & Run All**
- Wait ~2-3 minutes for complete execution

---

## Option 2: Run Prediction Script Only

If you just want to see predictions (model already trained):

```powershell
python src/predict.py
```

**Note:** This requires a pre-trained model at `models/titanic_survival_model.pkl`  
Run the notebook first to generate the model file.

---

## Option 3: Step-by-Step Execution

### 1. Set Up Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```powershell
jupyter notebook notebooks/titanic_ml_project.ipynb
```

### 3. Execute Cells Sequentially
- Read explanations in markdown cells
- Run code cells one by one (Shift + Enter)
- Observe outputs and visualizations

### 4. Save the Trained Model
- Model will be saved to `models/titanic_survival_model.pkl`
- This happens in Section 11 of the notebook

### 5. Test Predictions
```powershell
python src/predict.py
```

---

## 📁 Project Structure After Setup

```
mlpro/
│
├── data/                         # Dataset (auto-downloaded via seaborn)
│   └── titanic.csv              # Loaded in notebook
│
├── notebooks/                    # Jupyter notebooks
│   └── titanic_ml_project.ipynb # Main ML pipeline ✅
│
├── src/                          # Source code
│   └── predict.py               # Inference script ✅
│
├── models/                       # Trained models
│   └── titanic_survival_model.pkl  # Saved after running notebook
│
├── venv/                         # Virtual environment (if created)
│
├── requirements.txt             # Dependencies ✅
├── README.md                    # Documentation ✅
├── INTERVIEW_PREP.md            # Interview Q&A ✅
├── QUICK_START.md               # This file ✅
└── .gitignore                   # Git ignore ✅
```

---

## 🎯 Expected Outputs

After running the full notebook, you should see:

### Visualizations (7 total):
1. Missing values heatmap
2. Survival distribution (bar + pie chart)
3. Survival by gender (count + rate)
4. Survival by passenger class
5. Age distribution (overall + by survival)
6. Model performance comparison (bar chart)
7. Confusion matrix for best model

### Model Files:
- `models/titanic_survival_model.pkl` (saved pipeline)

### Metrics:
- Random Forest: ~82% accuracy
- F1-Score: ~77%
- Precision: ~81%
- Recall: ~74%

---

## ⚠️ Troubleshooting

### Issue: "No module named 'sklearn'"
**Fix:**
```powershell
pip install scikit-learn
```

### Issue: "Jupyter not found"
**Fix:**
```powershell
pip install jupyter
```

### Issue: Kernel won't start
**Fix:**
```powershell
python -m ipykernel install --user
```

### Issue: seaborn can't load dataset
**Fix:**
The notebook uses `sns.load_dataset('titanic')` which downloads data automatically.
If it fails, manually download from: https://github.com/mwaskom/seaborn-data

### Issue: Model file not saving
**Fix:**
- Ensure `models/` directory exists
- Check write permissions
- Verify Section 11 of notebook executed successfully

---

## 📊 What You'll Learn

By completing this project, you'll understand:

✅ **Full ML Pipeline**: Problem → Data → Model → Deployment  
✅ **EDA**: Visualizations and statistical analysis  
✅ **Preprocessing**: Missing values, encoding, scaling  
✅ **Feature Engineering**: Creating meaningful features  
✅ **Model Comparison**: Training multiple algorithms  
✅ **Hyperparameter Tuning**: GridSearchCV optimization  
✅ **Evaluation**: Accuracy, precision, recall, F1-score  
✅ **Production**: Pipelines and model deployment  

---

## 🎤 Interview Ready?

After completing the notebook:

1. **Read** `README.md` for project overview
2. **Study** `INTERVIEW_PREP.md` for 26 interview questions
3. **Practice** explaining your code out loud
4. **Customize** resume bullet points from README

---

## 🚀 Next Steps

### For Learning:
- Modify hyperparameters and observe changes
- Try different imputation strategies
- Add new engineered features (e.g., extract titles from names)

### For Portfolio:
- Push to GitHub with clear README
- Add to LinkedIn projects section
- Include in resume with metrics

### For Deployment:
- Create Flask/FastAPI REST API
- Containerize with Docker
- Deploy to AWS/Heroku

---

## 📞 Need Help?

If you encounter issues:
1. Check error messages carefully
2. Google the specific error
3. Refer to documentation:
   - Scikit-Learn: https://scikit-learn.org/
   - Pandas: https://pandas.pydata.org/
   - Seaborn: https://seaborn.pydata.org/

---

**Happy Learning! 🎓**
