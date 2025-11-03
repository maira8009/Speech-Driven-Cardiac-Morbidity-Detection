# Speech-Driven-Cardiac-Morbidity-Detection


### **Repository Name**

`Speech-Driven-Cardiac-Morbidity-Detection`

---

### **Short Description**

Speech and ECG fusion using diffusion probabilistic models for non-invasive cardiac morbidity detection and analysis.

---

### **README.md**

```markdown
# Speech-Driven Cardiac Morbidity Detection Using ECG Signals and Diffusion Models

## Overview
This repository contains the implementation, data preprocessing pipeline, and analysis scripts for the thesis **"Speech-Driven Cardiac Morbidity Detection Using ECG Signals and Diffusion Models"** by **Maira Hashmi** (MS Data Science, FAST School of Management, Islamabad). The project explores the feasibility of mapping speech biomarkers to ECG signals for cardiac abnormality detection using multimodal AI frameworks.

The study integrates **speech-derived acoustic features (MFCCs, jitter, shimmer, entropy)** with **ECG signal features (R-peaks, HRV, morphology)** and leverages **diffusion probabilistic models (DPMs)** for generative modeling and multimodal feature learning.

---

## Objectives
1. Develop a **speech-to-ECG mapping framework** using deep generative models.
2. Evaluate the **diagnostic performance** of diffusion-based models against classical ML baselines (SVM, Random Forest, MLP).
3. Validate the system’s ability to detect **multiple cardiac conditions** (heart failure, arrhythmia, atrial fibrillation).

---

## Repository Structure
```

Speech-Driven-Cardiac-Morbidity-Detection/
│
├── data/
│   ├── ECG_features_clean.csv
│   ├── final_speech_features.csv
│   ├── processed_speech_features.csv
│
├── notebooks/
│   └── analysis_notebook.ipynb     # Complete Jupyter Notebook with results & plots
│
├── scripts/
│   └── analysis_script.py          # Python version of notebook for automation
│
├── results/
│   ├── figures/                    # Generated ECG, ROC, and feature correlation plots
│   └── metrics/                    # Accuracy, AUC, and model comparison tables
│
├── README.md
└── requirements.txt

````

---

## Data
The analysis uses:
- **MIT-BIH Arrhythmia Database (ECG)** from PhysioNet
- **Mozilla Common Voice Dataset (Speech)** for acoustic features
- An **author-provided paired dataset** aligning speech and ECG samples

Each dataset is preprocessed for noise removal, normalization, and alignment.  
Feature extraction uses `librosa`, `neurokit2`, and `praat-parselmouth`.

---

## Methodology
### **1. Preprocessing**
- Speech: Noise filtering, silence removal, MFCC, jitter, shimmer, entropy extraction.
- ECG: Band-pass filtering, R-peak detection, HRV and QRS morphology computation.
- Features standardized and aligned temporally.

### **2. Modeling**
- Classical ML baselines: SVM (RBF), Random Forest, MLP.
- Generative Model: **Diffusion Probabilistic Model (DPM)** for ECG reconstruction from speech embeddings.

### **3. Evaluation**
Metrics include:
- **Accuracy**, **Sensitivity**, **Specificity**, **AUC**, **SSIM**  
- ROC and waveform similarity plots are generated per model.

---

## How to Run

### **1. Clone the repository**
```bash
git clone https://github.com/<yourusername>/Speech-Driven-Cardiac-Morbidity-Detection.git
cd Speech-Driven-Cardiac-Morbidity-Detection
````

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**

```bash
jupyter notebook notebooks/analysis_notebook.ipynb
```

### **4. Or execute the Python script**

```bash
python scripts/analysis_script.py
```

---

## Dependencies

List of core libraries:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
librosa
neurokit2
praat-parselmouth
torch
diffusers
```

---

## Results Summary

* Diffusion-based models outperform SVM and MLP in AUC and waveform reconstruction fidelity.
* Demonstrates significant correlation between **speech biomarkers** and **cardiac electrophysiological signals**.
* Highlights diffusion models as stable, physiologically interpretable frameworks for multimodal biomedical AI.

---

## Author

**Maira Hashmi**
MS Data Science, FAST School of Management
Supervisor: **Dr. Akhtar Jamil**

---

## License

This project is released under the MIT License.

---

Would you like me to also generate a `requirements.txt` file (exact Python packages and versions) based on your analysis environment? This will make your repo directly runnable by others.
