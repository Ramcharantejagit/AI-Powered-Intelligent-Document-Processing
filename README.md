<img width="1920" height="1080" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/f26733b0-9a81-439d-aa98-e76a4e41f3de" />
<img width="1920" height="1080" alt="Screenshot (13)" src="https://github.com/user-attachments/assets/24b0cfb5-6ab8-4a92-b63c-f2009a09fec7" />
<img width="1920" height="1080" alt="Screenshot (14)" src="https://github.com/user-attachments/assets/696d6812-55cc-4bd9-ab26-9d47490a19f2" />
# AI-Powered Intelligent Document Processing: Form Understanding on FUNSD

## Problem statement
Enterprises receive large volumes of scanned forms, such as invoices, fax cover sheets, claims and applications. Rule-based and manual processing does not scale.
This project builds an end-to-end **Intelligent Document Processing (IDP)** pipeline. It takes a scanned form, reads its words (OCR) and labels every word as part of a **HEADER**, **QUESTION** (key) or **ANSWER** (value), or as other text. It then extracts the entities and key-value pairs as structured JSON.
In machine-learning terms this is **semantic entity labelling**: multimodal sequence labelling over the OCR words, using text, layout and image cues, with BIO tags.

## Dataset
**FUNSD** (Form Understanding in Noisy Scanned Documents, Jaume et al., ICDAR-OST 2019), from https://guillaumejaume.github.io/FUNSD/
- 199 real, noisy scanned forms (149 train / 50 test), with word text, word boxes, entity labels and question-answer links
- After cleaning: 21,888 training words and 8,707 test words (empty OCR words removed)
- Details are in [`data/README.md`](data/README.md)

## IEEE paper
*AI-Powered Intelligent Document Processing in Digital Enterprises* ([ResearchGate](https://www.researchgate.net/publication/410824245_AI-Powered_Intelligent_Document_Processing_in_Digital_Enterprises)).
The paper surveys AI-based IDP systems from before 2019. It describes:
- a modular architecture: **ingestion → preprocessing (OCR) → classification → extraction → validation → storage**
- the use of supervised machine learning for classification and data extraction
- evaluation by **accuracy, precision, recall and processing time**

This project implements that architecture and uses the paper's classical machine-learning approach as the **existing system**.

## Proposed system
The proposed system keeps the paper's pipeline and replaces the classical classifiers with **advanced multimodal deep-learning models**.
All of them share one multimodal token encoder:
- **Text:** word embedding plus a character-level CNN, so unseen words still get a representation
- **Layout:** 30 scaled geometric and context features (position, size, gaps, line structure, colon cues)
- **Vision:** a CNN over each word's image crop, plus ink density, which captures bold and large headers

Every model decodes with a **linear-chain CRF** over the reading order, so the BIO tags it outputs are consistent.

## Algorithms
| System | Model | Description |
|---|---|---|
| Existing (paper) | Logistic Regression | Hand-crafted text, layout, visual and ±2-word context features → DictVectorizer → MaxAbsScaler → LR |
| Existing (paper) | Linear SVM | Same features; LinearSVC with sigmoid calibration to give probabilities |
| Existing (paper) | Random Forest | Same features; 300 trees |
| Existing (paper) | CRF | Linear-chain CRF (L-BFGS, elastic net) on the same features |
| Proposed | BiLSTM-CRF | Multimodal encoder → 2-layer BiLSTM over reading order → CRF |
| Proposed | LayoutGAT-CRF | Multimodal encoder → k-NN spatial graph (k=12) → 3 graph-attention layers with relative-geometry edge features → CRF |
| Proposed | SpatialTransformer-CRF | Multimodal encoder + continuous 2-D box encoding → 3-layer Transformer with learned relative-geometry attention bias → CRF |
| Proposed | **Graph-BiLSTM-CRF** (hybrid) | Multimodal encoder → 2 GAT layers (2-D context) → BiLSTM (reading-order context) → CRF |

Post-processing after classification:
- BIO tags are grouped into entities with a confidence score each.
- A rule-based step pairs each answer with its question (key-value pairs).
- Entities with low confidence are flagged for human validation.

**Training protocol**
- 15% of the training forms (22 documents) are held out as a validation set. Hyper-parameters, early stopping and the choice of best model all use **validation entity F1**.
- The 50 official test forms are used only for final reporting.
- Vocabularies and the numeric scaler are fit on the training split only, so there is no leakage.
- Word order is reconstructed from geometry alone, never from the annotation order.

> Pretrained document models such as LayoutLM could not be downloaded in the build environment, so every deep model is **trained from scratch** on CPU.

## Folder structure
```text
PROJECT_MAIN/
├── code/
│   ├── 01_Data_Preprocessing_and_Model_Training.ipynb   # EDA, cleaning, features, 8 models, evaluation, saving
│   ├── 02_Best_Model_Prediction.ipynb                   # independent inference with models/best_model only
│   └── idp_utils.py                                     # shared preprocessing, model classes, metrics, inference
├── data/
│   ├── raw/dataset/{training_data,testing_data}/{images,annotations}
│   ├── processed/        # cleaned word tables, split ids, prediction JSONs
│   ├── sample_input/     # sample new form for the prediction notebook
│   └── README.md
├── models/
│   ├── logisticregression/  linearsvm/  randomforest/  crf/                         # pipeline.joblib / crf_model.joblib + config.json + metrics.json
│   ├── bilstm_crf/  layoutgat_crf/  spatialtransformer_crf/  graph_bilstm_crf/      # model.pt + preprocessor.joblib + config.json + metrics.json + training_history.csv
│   ├── best_model/                 # copy of the selected model + best_model_info.json
│   └── model_evaluation_results/   # comparison CSVs, classification reports, confusion matrices, learning curves, EDA figures
├── webapp/                       # FormIQ Flask web application (see "Web application")
│   ├── app.py                    # routes, auth, RBAC, SQLite, IDP pipeline integration
│   ├── templates/  static/       # Bootstrap 5 + Chart.js UI (vendored, works offline)
│   └── instance/idp_app.db       # SQLite database (created automatically; uploads/ holds user files)
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
# Optional, for the raw-image OCR demo in notebook 02:
sudo apt-get install tesseract-ocr        # macOS: brew install tesseract
```
Tested with Python 3.13, torch 2.14 (CPU), scikit-learn 1.9.1, sklearn-crfsuite 0.5.0.

## How to run the training notebook
```bash
cd code
jupyter notebook 01_Data_Preprocessing_and_Model_Training.ipynb   # run all cells
# or headless:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_Data_Preprocessing_and_Model_Training.ipynb
```
- A full run takes about 40 minutes on a 2-core CPU. Most of that is the four deep models, each with up to 60 epochs and early stopping.
- It rewrites everything in `models/` and `data/processed/`.

## How to run the prediction notebook
```bash
cd code
jupyter notebook 02_Best_Model_Prediction.ipynb
```
- The notebook loads only `models/best_model/`. It does no training, and it does not use any variables from notebook 01.
- **Input 1:** a form image plus a words/boxes JSON. The sample is `data/sample_input/82092117.png` and `82092117_words.json`.
- **Input 2:** a raw image only. Words are extracted with Tesseract OCR.
- To use your own form, change `INPUT_IMAGE` / `INPUT_WORDS_JSON` or `RAW_IMAGE`.
- **Outputs:**
  - per-word labels with confidence
  - an annotated image
  - entities and key-value pairs
  - entities flagged for review
  - a JSON record in `data/processed/predictions/`

## Web application (FormIQ)
A Flask web app that puts the trained best model behind authenticated, role-based pages.

**What it does**
- It loads `models/best_model` once and runs the same `idp_utils` pipeline as the prediction notebook. The test suite checks that it returns identical predictions.
- **Inputs:** process a real FUNSD test form (using the dataset's OCR words, or Tesseract), or upload your own scanned form image, with an optional OCR words JSON.
- **Outputs per document:**
  - an annotated image
  - entities and key-value pairs
  - confidence scores, with entities under 60% flagged for review
  - processing time
  - a downloadable JSON file
  - for FUNSD test forms, a score against the gold annotation

**Pages**
- **Public:** Home, About (real model-comparison results), User Login, Admin Login, Register.
- **User:** dashboard with metrics and charts from the user's own runs, document processing, record details, history, profile (change password), logout.
- **Admin:** dashboard with pending approvals and charts; manage users (approve, reject or revoke, delete); view all records and all user activity.

**Security**
- Passwords are hashed with werkzeug scrypt.
- Registration always creates `role=user, status=PENDING`. Pending users see a pending message, and rejected users cannot log in. Role and approval are re-checked on every request, so revoking a user ends their session immediately.
- Users can only see their own records; other users' records return 404. Admin pages and APIs are enforced on the server.
- Other protections:
  - CSRF tokens on every POST
  - HttpOnly / SameSite session cookies with a 2-hour lifetime
  - login throttling (5 failures per 10 minutes)
  - input validation
  - checks on upload type and size (max 10 MB)
  - security headers

### Setup and run
```bash
pip install -r requirements.txt          # includes flask
sudo apt-get install tesseract-ocr       # needed only for OCR on uploaded images (macOS: brew install tesseract)
cd webapp
flask --app app init-db                  # optional: the app also creates the DB + admin on first start
python app.py                            # http://127.0.0.1:5000  (HOST / PORT env vars to change)
```
- **Default admin:** username `admin`, password `Admin@123`. It is created automatically in `webapp/instance/idp_app.db`. Change it after the first login (Profile page).
- **Optional environment variables:**
  - `IDP_SECRET_KEY` (otherwise a random key is generated in `instance/secret_key`)
  - `IDP_SECURE_COOKIES=1` when served over HTTPS
  - `IDP_INSTANCE_DIR` to put the DB and uploads elsewhere

**Test flow to try:** Register → see the pending message → Admin Login → approve the user → User Login → Dashboard → process a FUNSD test form → open the record → History. Then, as admin, check Records and Activity.

**Known limitations**
- Inference runs on CPU, one document at a time: about 0.1 s with supplied words, 2-5 s with Tesseract OCR.
- Tesseract output on these noisy scans is imperfect, so OCR-based runs cannot be scored against the gold labels.
- The Flask built-in server is for local or demo use. Use a WSGI server such as gunicorn with HTTPS for production.

## Model results
Actual results from the executed notebook, on the **50 held-out FUNSD test forms**:
- **Entity F1** is exact span and type match over HEADER / QUESTION / ANSWER.
- **Word** metrics cover 4 classes (other, header, question, answer).
- **ROC-AUC** is one-vs-rest, macro-averaged.

| Model | System | Val entity F1 | Test entity F1 | Entity P | Entity R | Word acc. | Word macro F1 | ROC-AUC | ms / doc |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Existing | 0.516 | 0.455 | 0.390 | 0.545 | 0.647 | 0.616 | 0.851 | 11 |
| Linear SVM | Existing | 0.519 | 0.465 | 0.402 | 0.552 | 0.648 | 0.608 | 0.858 | 14 |
| Random Forest | Existing | 0.529 | 0.499 | 0.448 | 0.564 | 0.655 | 0.622 | 0.871 | 89 |
| CRF | Existing | 0.583 | 0.565 | 0.639 | 0.506 | 0.626 | 0.599 | 0.840 | 12 |
| BiLSTM-CRF | Proposed | 0.722 | 0.690 | 0.708 | 0.673 | 0.720 | 0.701 | 0.875 | 53 |
| LayoutGAT-CRF | Proposed | 0.694 | 0.648 | 0.636 | 0.660 | 0.722 | 0.694 | 0.889 | 47 |
| SpatialTransformer-CRF | Proposed | 0.634 | 0.597 | 0.568 | 0.628 | 0.637 | 0.621 | 0.827 | 47 |
| **Graph-BiLSTM-CRF** | Proposed | **0.735** | **0.703** | 0.703 | 0.704 | **0.723** | **0.706** | **0.890** | 50 |

Entity F1 per class for the best model on the test set:

| Class | Entity F1 |
|---|---|
| Header | 0.489 |
| Question | 0.737 |
| Answer | 0.686 |

Full tables, confusion matrices and plots are in `models/model_evaluation_results/`.

## Best model
**Graph-BiLSTM-CRF** (hybrid graph attention + BiLSTM + CRF), selected on validation entity F1 (0.735).
- On the test set it scores entity F1 **0.703**, word accuracy **0.723** and ROC-AUC **0.890**.
- That is **+13.9 entity-F1 points** over the best existing-system model (CRF, 0.565) and +20.4 points over Random Forest.
- It is saved in `models/best_model/`: `model.pt`, `preprocessor.joblib`, `config.json`, `metrics.json` and `best_model_info.json`.

## Conclusion
- The project implements the paper's modular IDP pipeline on real scanned forms, end to end: OCR/ingestion, cleaning, reading order, feature engineering, classification, entity and key-value extraction, confidence-based validation and JSON storage.
- Every proposed multimodal deep model outperforms every classical existing-system model on entity-level F1.
- The best results come from combining 2-D spatial reasoning (graph attention), 1-D reading-order context (BiLSTM) and structured CRF decoding. This costs more processing time: about 50 ms per form against about 12 ms for the linear models, which is still fast enough for production.

**Limitations**
- All deep models were trained from scratch on 127 forms.
- Headers are rare and remain the hardest class.
- Key-value linking is rule-based.

**Future work**
- Fine-tune pretrained layout models (LayoutLMv3 / LiLT) when they can be downloaded.
- Add a learned relation-extraction head for question-answer linking.
