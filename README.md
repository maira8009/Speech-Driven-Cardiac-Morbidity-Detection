# Speech-Driven-Cardiac-Morbidity-Detection


### **Repository Name**

`Speech-Driven-Cardiac-Morbidity-Detection`

---

### **Short Description**

Speech and ECG fusion using diffusion probabilistic models for non-invasive cardiac morbidity detection and analysis.# Multimodal ECG and Speech Feature Fusion

## Project Overview
This repository contains the implementation of my MS thesis work focused on multimodal analysis using ECG and speech data for cardiac anomaly detection.

The core objective of this research is to study how **speech-based features** and **ECG-based physiological features** can be fused at the feature level even when the datasets belong to **different (unpaired) patients**.

---


This work addresses that gap by proposing a **feature-level fusion framework
for unpaired multimodal data**.



_________________________________________________Datasets_________________________________________
------ ECG Dataset----------
1:PTB-XL ECG dataset
2:Used for extracting physiological cardiac features

### Speech Dataset
- Speech Dataset fromthe Author of M. Usman, Z. Ahmad and M. Wajid, "Dataset of Raw and Pre-processed Speech Signals, Mel Frequency Cepstral Coefficients of Speech and Heart Rate Measurements," 2019 5th International Conference on Signal Processing, Computing and Control (ISPCC)
- Used for extracting MFCC and speech-related features

⚠️ **Important**: The ECG and speech datasets are from **different individuals** and are not patient-aligned.

---

## Feature Extraction
### ECG Features
- Heart Rate (HR)
- RR intervals
- R-peak statistics
- Time-domain statistical features

### Speech Features
- MFCC coefficients
- Mean and standard deviation of MFCCs
- Energy-based speech descriptors

---

## Feature Normalization
- StandardScaler is applied to normalize speech features
- ECG features are normalized separately
- This ensures equal contribution of both modalities during fusion

---

## Feature Fusion Technique
- **Feature-Level (Early) Fusion**
- Statistical aggregation (mean, std) is used
- Fused feature vectors are created without patient-level pairing



## Machine Learning Models
- Classical machine learning classifiers are used for evaluation
- Focus is on feasibility and methodological validation

---

## Why Diffusion Models Were Not Used
Although diffusion models are powerful for cross-modal generation, they were
not used in this thesis due to:
- Lack of paired multimodal medical data
- High computational cost
- Thesis focus on fusion feasibility rather than generative modeling

Diffusion models are identified as **future work**.

---


### **README.md**

```markdown
# Speech-Driven Cardiac Morbidity Detection Using ECG Signals and Diffusion Models

## Overview
This repository contains the implementation, data preprocessing pipeline, and analysis scripts for the thesis **"Speech-Driven Cardiac Morbidity Detection Using ECG Signals and Diffusion Models"** by **Maira Hashmi** (MS Data Science, FAST School of Management, Islamabad). The project explores the feasibility of mapping speech biomarkers to ECG signals for cardiac abnormality detection using multimodal AI frameworks.


---



---

## Repository Structure
```



````

---


---

## Author

**Maira Hashmi**
MS Data Science, FAST School of Management
Supervisor: **Dr. Akhtar Jamil**

---

## License

This project is released under the MIT License.

