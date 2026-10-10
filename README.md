# Alzheimer's Disease Classification & Prediction

> End-to-end machine learning classification project with EDA, feature engineering, model tuning, and a Streamlit prediction app.

## GitHub repository setup

- **Repository name:** `alzheimers-disease-classification`
- **Description:** `End-to-end Alzheimer's disease classification using EDA, feature engineering, model tuning, and a Streamlit prediction app.`
- **Suggested visibility:** Public only if you have permission to share the dataset and model artifacts; otherwise choose Private.
- **Topics:** `machine-learning`, `classification`, `scikit-learn`, `gradient-boosting`, `streamlit`, `python`, `data-science`, `eda`, `healthcare-analytics`
- **License:** Select one only after confirming redistribution rights for all included code, data, and artifacts.
- **Website:** Optional. Add a deployed Streamlit URL if you publish the app.

## Project overview

This project builds a binary classifier from the supplied Alzheimer's disease dataset. The workflow includes data inspection, exploratory data analysis (EDA), feature engineering, preprocessing, baseline model comparison, hyperparameter tuning, evaluation, model serialization, and Streamlit deployment.

**Educational/demo use only:** This model has not been clinically validated and must not be used as a medical diagnostic tool.

## Dataset

- **File:** `alzheimers_disease_data(1).xlsx`
- **Sheet:** `alzheimers_disease_data`
- **Size:** 2,149 rows × 35 original columns
- **Target:** `Diagnosis` (`0` = No Alzheimer's, `1` = Alzheimer's)

The supplied dataset had no missing values. `PatientID` was excluded as an identifier and `DoctorInCharge` was not included in the final model. Confirm you are allowed to redistribute the dataset before adding it to a public repository.

## Workflow

1. Load and inspect the data.
2. Perform EDA: missing values, distributions, target analysis, correlations, skewness, and outliers.
3. Engineer four additional features.
4. Split into train and test sets (80/20, `random_state=42`, stratified).
5. Preprocess continuous, binary, and categorical features.
6. Train and compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
7. Tune candidate models using `GridSearchCV` with 5-fold cross-validation and F1 scoring.
8. Evaluate the final model on the held-out test set.
9. Save the model and preprocessing artifacts using `joblib`.
10. Deploy inference through Streamlit.

## Feature engineering

- **`CognitiveSymptomCount`**: sum of `MemoryComplaints`, `BehavioralProblems`, `Confusion`, `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`, and `Forgetfulness` (range 0–7).
- **`CardiometabolicRiskCount`**: sum of `Smoking`, `CardiovascularDisease`, `Diabetes`, and `Hypertension` (range 0–4).
- **`PulsePressure`**: `SystolicBP - DiastolicBP`.
- **`Total_HDL_Ratio`**: `CholesterolTotal / CholesterolHDL`.

The final model input contains 36 features. `Ethnicity` and `EducationLevel` are treated as categorical despite being integer-coded.

## Preprocessing

- Continuous features: `StandardScaler`
- Binary features: passed through unchanged
- Categorical features: `OneHotEncoder(handle_unknown="ignore", drop="first")`
- Transformations are managed by a `ColumnTransformer`.
- The preprocessor is fitted during training; deployment uses `.transform()` only.

## Baseline model results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 94.42% | 92.11% | 92.11% | 92.11% | 93.89% |
| Gradient Boosting | 94.42% | 92.11% | 92.11% | 92.11% | 93.89% |
| Random Forest | 93.49% | 94.29% | 86.84% | 90.41% | 91.98% |
| Logistic Regression | 82.33% | 75.68% | 73.68% | 74.67% | 80.37% |

## Final model results

The selected model was the tuned **Gradient Boosting Classifier**.

| Metric | Held-out test result |
|---|---:|
| Accuracy | **95.58%** |
| Precision | **94.63%** |
| Recall | **92.76%** |
| F1 Score | **93.69%** |
| ROC-AUC | **94.94%** |

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 270 | 8 |
| Actual 1 | 11 | 141 |

These metrics describe the test set used in this project; they do not establish real-world diagnostic accuracy.

## Streamlit application

The interface collects demographic, lifestyle, medical history, cognitive/behavioral, blood pressure, cholesterol, and cognitive/functional assessment inputs. On prediction, the app calculates engineered features, constructs the expected 36-feature DataFrame, applies the saved preprocessor, runs the model, and displays the predicted class and class probabilities.

## Suggested repository structure

```text
alzheimers-disease-classification/
├── README.md
├── app.py
├── requirements.txt
├── final_gradient_boosting.pkl
├── preprocessor.pkl
├── deployment_info.pkl
├── model_summary.pkl
├── notebooks/
│   └── alzheimers_disease_classification.ipynb
├── data/
│   └── alzheimers_disease_data(1).xlsx
└── screenshots/
    ├── app_home.png
    ├── prediction_result.png
    └── engineered_features.png
```

Adjust this structure to match the files you actually have. Do not commit `venv/`, secrets, or unrelated local files. Large files may require Git LFS or an approved external storage location.

## Tech stack

Python · pandas · NumPy · scikit-learn · joblib · Streamlit · Jupyter Notebook · Git/GitHub

## Environment versions

The training environment reported:
- Python `3.13.9`
- scikit-learn `1.7.2`
- joblib `1.5.2`
- NumPy `2.2.6`

Serialized scikit-learn models should be loaded with compatible package versions.

## Installation and run

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Create `requirements.txt` with:

```text
streamlit
pandas
numpy==2.2.6
scikit-learn==1.7.2
joblib==1.5.2
```

Install and run:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Git commands

```bash
git init
git add README.md app.py requirements.txt
git status
git commit -m "Add Alzheimer's disease classification project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Suggested `.gitignore`:

```gitignore
venv/
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.streamlit/secrets.toml
.env
.DS_Store
```

## Limitations

- No independent external validation dataset was used.
- The model has not been clinically validated.
- Dataset representativeness and bias may affect generalization.
- Correlation does not imply causation.
- Model probabilities are outputs of the trained classifier, not clinically calibrated patient risk.

## Future improvements

- External validation and broader cross-validation
- Probability calibration and threshold analysis
- Feature importance and SHAP explainability
- Additional model comparisons
- Input validation and model/data drift monitoring
- Docker packaging and cloud deployment
- Clinical expert review before any real-world medical use

## Author

- **Name:** Replace with your preferred name
- **GitHub:** Add your GitHub profile URL
- **LinkedIn:** Add your LinkedIn URL (optional)
- **Live demo:** Add your deployed application URL (optional)

## Disclaimer

This repository is for educational and demonstration purposes only. The model is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or make healthcare decisions. Consult qualified healthcare professionals for medical concerns.
