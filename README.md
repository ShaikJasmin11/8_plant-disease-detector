#  Plant Disease Detector (CNN)

> RISE Internship Project 8 – Tamizhan Skills  
> Built with TensorFlow, Keras, CNN, and Streamlit

A deep learning-based web app that detects plant leaf diseases from images using a Convolutional Neural Network (CNN). This is the eighth and final project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build a plant disease detection model that:
  - Loads plant disease images organized into class folders
  - Applies image preprocessing and augmentation
  - Trains a **CNN classifier** on 16 plant disease classes
  - Provides an intuitive **Streamlit interface** for image upload and prediction

---

##  Tech Stack

- **Python**
- **TensorFlow / Keras (CNN)**
- **NumPy / Matplotlib**
- **Pillow (image handling)**
- **Streamlit** (for frontend UI)

---

##  Project Structure

```bash
plant-disease-detector/
├── app.py                        # Streamlit frontend for prediction
├── main.py                       # CNN model training script
├── requirements.txt              # All required packages
├── data/
│   └── PlantVillage/             # Folder with 16 subfolders (classes)
├── models/
│   └── plant_disease_model.h5    # Trained Keras model
├── src/
│   └── preprocess.py             # Data loading & image augmentation
└── README.md                     # You're reading it 😉
```

---

## Dataset

- Source: PlantVillage Dataset – Kaggle(https://www.kaggle.com/datasets/emmarex/plantdisease)
- Structure: Folder-per-class format (e.g., Tomato___Early_blight, Pepper__bell___Bacterial_spot, etc.)
- Contains ~33,000+ images across 16 plant disease categories
- Augmented during training using ImageDataGenerator

---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Trained on 33k+ images across 16 plant disease classes
✅ Achieved >50% accuracy in 3 epochs (base CNN, no pretraining)
✅ Uses real-time inference on uploaded leaf images
✅ Supports deployment on CPU with lightweight image sizes (64×64)

---

## Highlights

- Supports image-based classification of plant diseases
- Modular CNN built with Dropout and MaxPooling for generalization
- Custom training loop with early stopping & learning rate scheduling
- Streamlit interface for image upload and prediction
- Easily extensible to new plant diseases or camera integration
  
---

## Future Improvements

- Integrate pretrained MobileNetV2 for better accuracy & speed
- Add confidence scores and probability bar chart
- Display leaf image preview with prediction
- Export predictions to PDF/CSV for field reports

---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world challenges in agriculture, this project aims to empower farmers and researchers with accessible, AI-powered disease diagnosis.

Built by @ShaikJasmin11
