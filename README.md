# 🏃 Human Action Detection — Transfer Learning

A computer vision project that classifies human actions from images using **Transfer Learning** with pre-trained deep CNN models. The model learns to recognize a variety of human activities by fine-tuning a powerful backbone trained on ImageNet.

---

## 📌 Overview

Manually labeling or monitoring human activities in video/image feeds is time-consuming and error-prone. This project automates the process using deep learning — specifically Transfer Learning — to build a high-accuracy image classifier capable of identifying what action a person is performing in a given image.

---

## 🏷️ Action Categories

The model is trained to recognize common human activities such as:

| Category | Category | Category |
|---|---|---|
| Calling | Clapping | Cycling |
| Dancing | Drinking | Eating |
| Fighting | Hugging | Laughing |
| Listening to Music | Running | Sitting |
| Sleeping | Texting | Using Laptop |

> *Exact classes depend on the dataset used. Update this table to match your data.*

---

## 🧠 Approach

1. **Data Loading & Exploration** — Visualizing class distributions and sample images
2. **Preprocessing** — Image resizing, normalization, and augmentation (flipping, rotation, zoom)
3. **Transfer Learning** — Loading a pre-trained CNN backbone (e.g., VGG16 / ResNet / MobileNet / InceptionV3) with ImageNet weights
4. **Fine-Tuning** — Freezing base layers, replacing the classification head, and training on the target dataset
5. **Evaluation** — Accuracy, loss curves, confusion matrix, and per-class metrics

---

## 📁 Project Structure

```
Human_Action_Detaction/
├── Human_Action_Detaction_Transfer_learning.ipynb   # Main notebook
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| Deep Learning | PyTorch |
| Transfer Learning | Pre-trained CNN ResNet18 |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JenilGoti/Human_Action_Detaction.git
cd Human_Action_Detaction
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

Or if using PyTorch:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the Notebook

```bash
jupyter notebook Human_Action_Detaction_Transfer_learning.ipynb
```

---

## 📊 Dataset

This project uses the **Human Action Recognition (HAR)** dataset available on Kaggle.

1. Download the dataset from [Kaggle — Human Action Recognition](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset)
2. Extract and place it in a `data/` folder:

```
Human_Action_Detaction/
└── data/
    ├── train/
    │   ├── calling/
    │   ├── cycling/
    │   └── ...
    └── test/
        ├── calling/
        ├── cycling/
        └── ...
```

---

## 🚀 How It Works

```
Input Image
     ↓
Preprocessing (resize → normalize → augment)
     ↓
Pre-trained CNN Backbone (frozen weights)
     ↓
Custom Classification Head (Dense + Softmax)
     ↓
Predicted Action Label
```

The key idea behind Transfer Learning is reusing the feature extraction layers of a model already trained on millions of images (ImageNet), then training only the final classification layers on the action dataset — saving significant compute time while achieving high accuracy.

---

## 📈 Results


---

## 🔮 Future Improvements

- Extend to **video-based action recognition** using LSTMs or 3D CNNs
- Deploy as a **real-time webcam app** using OpenCV
- Experiment with **Vision Transformers (ViT)** as the backbone
- Add **Grad-CAM visualizations** to interpret model predictions

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---
