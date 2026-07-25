# DeepSkinDiag-XAI-An-Explainable-Multimodal-AI-System-for-Web-Based-Skin-Disease-Diagnosis


DeepSkinDiag-XAI is a Progressive Web Application (PWA) that assists in skin disease diagnosis by combining image-based and symptom-based analysis using multimodal artificial intelligence. The system integrates EfficientNet for skin image classification and DistilBERT for symptom understanding, then combines both predictions through a Late Fusion mechanism to generate more reliable diagnostic results.

To promote transparency and trust, the platform incorporates Explainable AI (XAI) techniques such as Grad-CAM visualizations and confidence calibration, allowing users to understand why a prediction was made rather than receiving a black-box result.

The application supports both English and Sinhala, making AI-powered skin health guidance more accessible to a wider audience.

---

## Features

* Multimodal diagnosis using skin images and symptom descriptions
* EfficientNet-based skin disease image classification
* DistilBERT-based symptom text analysis
* Late Fusion architecture for improved prediction accuracy
* Explainable AI using Grad-CAM heatmaps
* Confidence score calibration
* Responsive Progressive Web Application (PWA)
* Bilingual interface (English & Sinhala)
* Mobile and desktop compatibility
* User-friendly diagnostic reports

---

## System Architecture

```text
User Input
├── Skin Image
│   └── EfficientNet Model
│
└── Symptom Description
    └── DistilBERT Model

Predictions
      ↓
 Late Fusion Layer
      ↓
 Final Diagnosis
      ↓
 Explainability Module
 (Grad-CAM + Confidence Score)
```

---

## Screenshots

### Dataset Sample

Add a representative thermal skin image from the dataset.

```markdown
!<img width="616" height="584" alt="Screenshot 2025-10-21 at 10 12 12" src="https://github.com/user-attachments/assets/c75e488f-2533-4835-a4cf-9ba82eb90109" />
(images/dataset-sample.jpg)
```

### Mobile Application

| HomReal time test by camara
<img width="780" height="3156" alt="IMG_3587" src="https://github.com/user-attachments/assets/176eecb8-ddae-412b-91c4-aa28af123867" />


                           |
| Test by uploading photo
<img width="780" height="5796" alt="IMG_3591" src="https://github.com/user-attachments/assets/2e5f5f1c-b504-45e3-9b8c-b0bca10ca7a1" />


### Desktop Application

| Analysis Page         

<img width="841" height="818" alt="Screenshot 2025-10-21 at 12 15 55" src="https://github.com/user-attachments/assets/cd95629e-227a-4e08-aba3-4fb7ca845156" />


<img width="841" height="818" alt="Screenshot 2025-10-21 at 12 20 40" src="https://github.com/user-attachments/assets/cc9172ee-a17c-4b26-a556-b8251bf435d7" />


---

## Explainable AI (XAI)

DeepSkinDiag-XAI provides visual explanations through Grad-CAM heatmaps that highlight the image regions influencing model predictions.

Benefits include:

* Improved transparency
* Increased user trust
* Better clinical interpretability
* Easier debugging and model validation

---

## Technology Stack

### Frontend

* React.js
* Progressive Web App (PWA)
* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask / FastAPI

### Artificial Intelligence

* EfficientNet
* DistilBERT
* Late Fusion Architecture
* Grad-CAM
* Confidence Calibration

### Data Processing

* NumPy
* Pandas
* OpenCV
* Scikit-learn

## Dataset

This project utilizes multiple publicly available dermatology datasets:

- HAM10000
- ISIC Archive
- DermNet

Links:
- https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- https://challenge.isic-archive.com/
- https://dermnetnz.org/

---

## Workflow

1. User uploads a skin image.
2. User enters symptom descriptions.
3. EfficientNet analyzes the image.
4. DistilBERT analyzes textual symptoms.
5. Predictions are merged using Late Fusion.
6. Confidence calibration is applied.
7. Grad-CAM explanations are generated.
8. Final diagnosis and explanations are displayed.

---

## Research Contribution

This project demonstrates how multimodal AI and explainable machine learning can be combined to create a transparent and accessible skin disease diagnostic assistant.

Key contributions:

* Multimodal image-text diagnosis
* Explainable AI integration
* Confidence-aware predictions
* Bilingual healthcare support
* Web-based deployment through PWA technology

---

## Installation

```bash
git clone https://github.com/SanWithanage/DeepSkinDiag-XAI-An-Explainable-Multimodal-AI-System-for-Web-Based-Skin-Disease-Diagnosis.git

cd DeepSkinDiag-XAI-An-Explainable-Multimodal-AI-System-for-Web-Based-Skin-Disease-Diagnosis

pip install -r requirements.txt
```

---

## Running the Project

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

## Future Improvements

* Additional skin disease classes
* Dermatologist feedback integration
* Multi-language expansion
* Clinical validation studies
* Cloud deployment
* Real-time camera diagnosis

---

## Disclaimer

This project is intended for educational and research purposes only. It does not replace professional medical diagnosis, treatment, or consultation from qualified healthcare practitioners.

---

## Author

**San Withanage**

Undergraduate Research Project

Department of Computing & Data Science 

NSBM Green University of Sri Lanka

