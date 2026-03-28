
---

# 🦴 Bone Fracture Classification using Deep Learning

## 📌 Overview

This project focuses on building a deep learning model to classify different types of bone fractures from medical images. The system leverages Convolutional Neural Networks (CNNs) to automatically identify fracture types and assist in medical diagnosis.

The primary focus of this project is not only achieving model performance but also gaining a strong understanding of the **end-to-end Machine Learning pipeline and real-world challenges**.

---

## 🚀 Features

* 📊 Multi-class classification (10 fracture types)
* 🧠 Deep Learning-based image classification
* 🔄 Data preprocessing & augmentation using generators
* 📈 Training and validation pipeline
* ⚙️ Modular and scalable code structure

---

## 🧬 Dataset

The dataset consists of medical images categorized into the following fracture types:

* Avulsion fracture
* Comminuted fracture
* Fracture Dislocation
* Greenstick fracture
* Hairline Fracture
* Impacted fracture
* Longitudinal fracture
* Oblique fracture
* Pathological fracture
* Spiral Fracture

**Dataset Summary:**

* Training samples: **907**
* Validation samples: **222**

---

## 🏗️ Project Structure

```
project/
│── data/                  # Dataset directory
│── src/                   # Source code
│   ├── data_loader.py     # Data preprocessing & generators
│── notebooks/
│   ├── training.ipynb     # Model training notebook
│── models/                # Saved models
│── README.md              # Documentation
```

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy & Pandas
* Matplotlib / Seaborn
* Jupyter Notebook / VS Code

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bone-fracture-classification.git

# Navigate to project directory
cd bone-fracture-classification

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Prepare Dataset

Organize your dataset in the following structure:

```
data/
│── train/
│── validation/
```

### 2. Train the Model

```bash
python train.py
```

Or run the notebook:

```
notebooks/experimentation.ipynb
```

### 3. Output

* Trained model will be saved in `/models`
* Training metrics will be displayed during execution

---

## 📊 Model Workflow

1. Load dataset using data generators
2. Apply preprocessing & augmentation
3. Train CNN model
4. Validate model performance
5. Save trained model

---

## ⚠️ Challenges & Learnings

### 📉 Limited Dataset Size

A key challenge in this project was the **small dataset size**, which limited the model’s ability to generalize effectively.

* Approximately 1100 images across 10 classes
* Increased risk of overfitting
* Limited representation of real-world scenarios

---

### 📊 Low F1-Score & Prediction Limitations

Due to dataset constraints:

* The model achieved a relatively **low F1-score** for certain classes
* Imbalance between precision and recall
* Some fracture types were misclassified more frequently

---

### 🧠 Key Learnings

This project helped in understanding the complete ML workflow:

* Data preprocessing and loading
* Data augmentation techniques
* Model training and validation
* Performance evaluation (Accuracy, F1-score)
* Real-world challenges in ML systems

---

## 🎯 Project Objective

> The main goal of this project was to learn and implement the complete Machine Learning pipeline and understand practical challenges such as limited data, model generalization, and evaluation metrics.

---

## 🚀 Future Improvements

* Increase dataset size and diversity
* Apply class balancing techniques
* Use transfer learning (ResNet, EfficientNet)
* Add confusion matrix and class-wise evaluation
* Deploy using Streamlit or Flask

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request


---

