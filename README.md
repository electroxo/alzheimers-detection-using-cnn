# 🧠 Alzheimer's Disease Detection Using Deep Learning (FYP)

This repository contains the code for my Final Year Project (FYP): a machine learning and deep learning-based approach for detecting **Alzheimer’s Disease** using neuroimaging data and clinical features. The project involves multiple model architectures to compare performance and accuracy in diagnosing different stages of the disease.

---

## 👨‍💻 Author

**Shahzad Ahmed**  
🔗 [LinkedIn](https://www.linkedin.com/in/shahzad-ahmed-92427a27a/)  
📧 shahzadahmed9712@gmail.com  
🇵🇰 AI/ML Enthusiast | Nickname: `electro`  
Currently learning: **LLMs**, **Attention Mechanisms**, **Computer Vision**, and **Cyber Security**

---

## 📁 Project Structure

alzheimers-cnn/
├── models/
│ ├── custom_cnn.ipynb # Basic CNN for image classification
│ ├── efficientnet_b4.ipynb # Transfer learning using EfficientNet-B4
│ ├── ensemble_model.ipynb # Combines multiple predictions
│ └── xgboost_clinical_data.ipynb # XGBoost model on structured clinical data
├── results/ # Accuracy plots, confusion matrices, metrics
├── requirements.txt # Required Python packages
└── README.md # Project overview and documentation

yaml
Copy
Edit

---

## 🧪 Models Used

| Model                    | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| 🧠 **Custom CNN**         | A simple CNN architecture built from scratch using TensorFlow/Keras |
| 🌱 **EfficientNet-B4**    | Transfer learning model pre-trained on ImageNet for medical imaging |
| 🔗 **Ensemble Model**      | Combines CNN and EfficientNet predictions for improved accuracy |
| 📊 **XGBoost**             | Trained on structured clinical features (non-image data)     |

---

## 📊 Dataset

The dataset includes **MRI brain scans** and/or **clinical tabular data**.  
Each sample is categorized into stages of Alzheimer’s Disease (e.g., Non-Demented, Mild Demented, Moderate, etc.).

> *Note:* For privacy and size reasons, the dataset is not uploaded here. You can use publicly available datasets such as:
- [Kaggle: Alzheimer’s MRI Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)
- [OASIS Brain Dataset](https://www.oasis-brains.org/)

---

## ✅ Results Summary

| Model              | Accuracy | Notes                                |
|--------------------|----------|--------------------------------------|
| Custom CNN         | ~85%     | Good baseline performance            |
| EfficientNet-B4    | ~93%     | Strong results via transfer learning |
| Ensemble Model     | ~95%     | Best accuracy after combining models |
| XGBoost (Clinical) | ~87%     | Performed well on structured data    |

📌 *Accuracy may vary depending on preprocessing and hyperparameters.*

---


## 📷 Sample Outputs

Images like:
- Loss/Accuracy plots
- Confusion Matrices
- ROC curves

...can be found in the `/results/` folder (add them if available).

---

## 🛠️ Languages & Tools

- Python
- Jupyter Notebook
- TensorFlow / Keras
- scikit-learn
- XGBoost
- Pandas, NumPy, Matplotlib, Seaborn
- Google Colab / VS Code

---


## 📦 Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/electroxo/alzheimers-cnn.git
cd alzheimers-cnn
