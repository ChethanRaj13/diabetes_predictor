import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# Configuration
# -----------------------------
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_FILE = "file1.csv"

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"

CATEGORICAL_COLS = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]


# -----------------------------
# Data Loading
# -----------------------------
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_data(train_df, test_df, categorical_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Fit on training data only,
    train_cat = encoder.fit_transform(train_df[categorical_cols])
    test_cat = encoder.transform(test_df[categorical_cols])

    cat_feature_names = encoder.get_feature_names_out(categorical_cols)

    train_cat_df = pd.DataFrame(train_cat, columns=cat_feature_names)
    test_cat_df = pd.DataFrame(test_cat, columns=cat_feature_names)

    # Drop categorical columns
    train_df = train_df.drop(columns=categorical_cols)
    test_df = test_df.drop(columns=categorical_cols)

    # Concatenate encoded features
    train_df = pd.concat([train_df.reset_index(drop=True), train_cat_df], axis=1)
    test_df = pd.concat([test_df.reset_index(drop=True), test_cat_df], axis=1)

    return train_df, test_df


# -----------------------------
# Model Training
# -----------------------------
def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


# -----------------------------
# Main Workflow
# -----------------------------
def main():
    # Load data
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    # Separate target and ID
    y_train = train_df[TARGET_COL].copy()
    train_ids = train_df[ID_COL]
    test_ids = test_df[ID_COL]

    train_df = train_df.drop(columns=[TARGET_COL, ID_COL], axis=1)
    test_df = test_df.drop(columns=[ID_COL], axis=1)

    # Preprocess
    X_train, X_test = preprocess_data(train_df, test_df, CATEGORICAL_COLS)

    # Train
    model = train_model(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Save results
    output = pd.DataFrame({
        "id": test_ids,
        TARGET_COL: predictions
    })
    output.to_csv(OUTPUT_FILE, index=False)

    print("Prediction complete. Results saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
