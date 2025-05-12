# Eye Disease Detection Using Deep Learning

This project aims to classify eye diseases using deep learning techniques, specifically transfer learning with the **VGG19** model. It focuses on identifying four categories: **Normal**, **Cataract**, **Diabetic Retinopathy**, and **Glaucoma**.

---

## 🚀 Project Highlights

- 🔍 **Model Used**: Pre-trained VGG19 with added Dense layers.
- 🧠 **Classes**: Normal, Cataract, Diabetic Retinopathy, Glaucoma.
- 📊 **Performance**:
  - Test Accuracy: ~78.74%
  - Precision & Recall analyzed using classification report.
- 📉 **Loss Function**: Categorical Crossentropy
- ⚙️ **Optimizer**: Adam (Learning Rate = 0.0001)

---

## 🧬 Project Structure
![Screenshot 2025-05-12 153252](https://github.com/user-attachments/assets/e67b980a-8c4b-4177-a5b8-ac3c1111de21)

---

## 🧠 Methodology

1. **Data Preparation**:
   - Images resized to 224x224.
   - Augmentation: rotation, width/height shift, zoom.

2. **Model Architecture**:
   - Base: Frozen VGG19.
   - Custom Head: Flatten → Dense(512) → Dropout(0.5) → Dense(4, softmax).

3. **Training Strategy**:
   - Batch Size: 32
   - Epochs: 50
   - Validation and Test Split maintained.
   - EarlyStopping and ModelCheckpoint used.

4. **Evaluation**:
   - Classification Report
   - Confusion Matrix
   - Precision/Recall/F1 Score per class

---

##🧪 Results Summary
| Class                | Precision | Recall | F1-Score |
| -------------------- | --------- | ------ | -------- |
| Normal               | 0.63      | 0.83   | 0.72     |
| Cataract             | 0.81      | 0.78   | 0.79     |
| Diabetic Retinopathy | 0.83      | 0.77   | 0.80     |
| Glaucoma             | 0.71      | 0.60   | 0.65     |

##✅ Future Work
Fine-tune VGG19 layers for better feature extraction.

Use advanced models like EfficientNet or Vision Transformers.

Add attention mechanisms (e.g., CBAM).

Integrate Grad-CAM for visual explanation.

Address class imbalance with focal loss or oversampling.

Improve augmentation (CLAHE, color jittering).

##📚 References
VGG19: https://keras.io/api/applications/vgg/

TensorFlow Documentation: https://www.tensorflow.org/api_docs

Eye Disease Datasets: [[Insert Link Here](https://drive.google.com/drive/folders/1N7-SFq3oQSN_jgElbHwPj4vFaocTz8xn?usp=drive_link)]

👨‍💻 Author
[Abhay tiwari]

B.Tech 3rd Year, SRMS CET&R

Linkedin-[[Insert Link Here](https://www.linkedin.com/in/abhaytiwari30)]



