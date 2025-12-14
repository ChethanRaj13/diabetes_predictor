# Diabetes Predictor

This project trains a machine learning model to predict the likelihood of diabetes diagnosis using patient demographic, lifestyle, and health data. It handles categorical and numerical features, trains a Random Forest model, and generates predictions on unseen data.

The system automatically:
- Loads training and test datasets
- Applies one-hot encoding to categorical features
- Trains a Random Forest model
- Runs predictions on test data
- Saves the results as a CSV file

---

## 📁 Files in the Project

| File | Purpose |
|------|---------|
| `train.csv` | Raw dataset for training |
| `test.csv` | Test dataset for prediction |
| `diabetes_predictor.py` | Main script to train model and generate predictions |
| `file1.csv` | Final prediction output with predicted diabetes likelihood |

---

## ⚙️ How It Works

1. Load training and test datasets
2. Separate categorical and numerical features
3. Apply one-hot encoding to categorical features
4. Train a Random Forest Regressor
5. Make predictions on the test dataset
6. Save predictions into `file1.csv`

---

## ▶️ Running the Project

### 1. Install Dependencies
```bash
pip install pandas scikit-learn
```

##2. Execute the Script
```bash
python diabetes_predictor.py
```

After running, you should see file1.csv containing the test IDs and predicted diabetes probabilities.
