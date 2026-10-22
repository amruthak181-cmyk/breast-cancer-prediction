# Breast Cancer Prediction System

A machine learning project that predicts whether a breast tumour is **Benign** or **Malignant** using a Random Forest Classifier, with an interactive Streamlit web app.

**Live demo:** [add your Streamlit link here]

## Project Description
The model is trained on the Breast Cancer dataset from Kaggle (569 records). Users can adjust tumour measurements in the web app and get a prediction with a confidence score and a feature-importance chart.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Machine Learning Algorithm
Random Forest Classifier (100 trees, `random_state=42`), trained on an 80/20 train-test split.

## Features Used
- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Concave Points Mean
- Radius Worst
- Area Worst

## Results

| Metric | Score |
|---|---|
| Accuracy | 95.6% |
| Precision (malignant) | 95.2% |
| Recall (malignant) | 93.0% |
| F1 score (malignant) | 94.1% |

Test set: 114 samples (71 benign, 43 malignant).

## Screenshots
[Add a screenshot of your Streamlit app here]

## Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Files
- `main.py`: data checks, training, evaluation and charts
- `app.py`: Streamlit web app
- `breast-cancer.csv`: dataset
- `breast_cancer_model.pkl`: saved trained model

## Disclaimer
This project is for learning purposes only and is not a medical diagnosis tool.
