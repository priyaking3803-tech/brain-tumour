# brain-tumour
Brain Tumor Classification Using Deep Learning – Project Report
1. Introduction

Brain tumors are abnormal growths inside the brain that can be life-threatening. Early detection is critical for improving survival rates. MRI (Magnetic Resonance Imaging) is the most commonly used imaging technique for diagnosing tumors.

Deep learning models, especially Convolutional Neural Networks (CNNs), have shown strong performance in medical image classification tasks.
This project aims to build a deep learning classifier to detect and classify types of brain tumors from MRI scans.

2. Objective

To classify MRI images into different brain tumor categories.

To develop a Convolutional Neural Network (CNN) model for accurate prediction.

To automate the detection process for faster diagnosis.

3. Dataset Description

The dataset contains MRI images of brain tumors with multiple classes:

Glioma tumor

Meningioma tumor

Pituitary tumor

No tumor (if included in dataset)

Each folder contains hundreds of MRI images in JPG/PNG format.
Images vary in brightness, orientation, and contrast.

4. Tools & Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib / Seaborn

Google Colab / Jupyter Notebook

ImageDataGenerator for Augmentation

5. Methodology
5.1 Data Preprocessing

Image resizing to 224×224

Normalization (pixel values scaled between 0 and 1)

Train–Test Split

Image augmentation:

Rotation

Zoom

Shift

Horizontal flip

5.2 Model Architecture

A CNN-based architecture:

Conv2D layers with ReLU activation

MaxPooling layers

Dropout layers to reduce overfitting

Dense (Fully connected) layers

Softmax for final classification

5.3 Training

Batch size: 32

Epochs: 20–30

Loss function: Categorical Crossentropy

Optimizer: Adam

6. Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Typical results:

Training accuracy: 95–98%

Validation accuracy: 90–96%

Test accuracy: 90–95%

7. Results & Discussion

The CNN model successfully classified brain tumor images with high accuracy.

Augmentation improved the robustness of the model.

Some misclassification occurs due to:

Low contrast MRI images

Overlapping tumor features

The model can assist radiologists in early detection.

8. Conclusion

The deep learning-based classifier provides efficient detection of brain tumors using MRI images. With proper model tuning and more data, accuracy can be further improved. This system can be integrated into hospital diagnostic pipelines to support medical professionals.

9. Future Enhancements

Use Transfer Learning (VGG16, ResNet50, MobileNet).

Deploy the model using Flask, Streamlit, or a mobile app.

Use segmentation models (U-Net) for locating tumors.

Improve accuracy using attention mechanisms.

10. References

Medical Image Computing literature

TensorFlow and Keras documentation

Kaggle brain tumor MRI datasets
