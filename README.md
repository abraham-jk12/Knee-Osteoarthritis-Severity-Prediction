# Knee Osteoarthritis KL Grade Classification System

Automated Kellgren-Lawrence grading of knee osteoarthritis from X-ray radiographs using a 3-model deep learning ensemble with Test Time Augmentation.

## Results
- Final QWK: 0.8367 (exceeds human radiologist inter-rater agreement of 0.737–0.818)
- Accuracy: 68.14%
- Models: DenseNet-169 + EfficientNet-B5 + EfficientNet-V2-S

## Project Structure
- `app_updated.py` — Flask web application
- `evaluate_app.py` — Full evaluation script (10 methods)
- `optimize_ensemble.py` — Scipy weight optimization
- `split_and_sort_knees.py` — Bilateral image splitting script
- `templates/index.html` — Web interface
- `training_scripts/` — Individual model training scripts

## Setup
Install dependencies:
```
pip install -r requirements.txt
```

## Data
This project uses X-ray images from the Osteoarthritis Initiative (OAI).
Data access requires institutional approval at: https://nda.nih.gov/oai
Package used: OAIScreeningImages (Package ID: 1244889)
Labels: kxr_sq_bu01.txt

Due to the OAI data use agreement, images cannot be shared publicly.

## Model Weights
Model weights cannot be included in this repository due to file size.
Train your own models using the scripts in training_scripts/

## Running the App
Once you have the model weights and data:
```
python app_updated.py
```
Then open http://127.0.0.1:5000 in your browser.
