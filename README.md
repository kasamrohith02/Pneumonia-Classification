# Real-Time Pneumonia Classification Using Machine Learning

## Dataset

- **Name**: Pneumonia X-Ray Images
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images)
- **Volume**: 5,856 images

## Flow Diagram

Below is the flow diagram illustrating the process for pneumonia detection:

![Flow Diagram](./Flow%20Diagram.png)

## Preprocessing Techniques

- Rescaling, zooming, and horizontal flipping
- Grayscale conversion

## Machine Learning Models

- **Model Used**: Convolutional Neural Network (CNN)
  - Early stopping was implemented to avoid overfitting.
  - `ReduceLROnPlateau` was used as a callback to reduce the learning rate when a metric stops improving.
  - Handled imbalance issues using `sklearn.utils.compute_sample_weight`.
  - Plots were generated to visualize accuracy and loss for both training and testing data.

### Metrics Used

- Accuracy
- F1 Score
- Confusion Matrix
- Precision
- Recall

- A confusion matrix was plotted to evaluate the model's performance.
- Validation loss was monitored during training. If the validation loss did not improve for two consecutive epochs, the learning rate was reduced by a factor of 0.3.

## Results

- **Test Accuracy**: 91.51%

**Detailed Results**:
```
              precision    recall  f1-score   support

      NORMAL       0.96      0.81      0.88       234
   PNEUMONIA       0.89      0.98      0.94       390

    accuracy                           0.92       624
   macro avg       0.93      0.89      0.91       624
weighted avg       0.92      0.92      0.91       624
```

## Deployment

- The model was deployed using **Streamlit**.
- The web page takes an image as input and outputs the class to which it belongs, along with the confidence percentage.

### How It Works

1. **Image Loading**: Loads an image from a file.
2. **Image Preprocessing**:
   - Resizes the image to a target size of 500x500 pixels.
   - Converts the image to grayscale.
   - Converts the image to a numpy array and normalizes pixel values to the range [0, 1].
3. **Prediction**: 
   - Uses the trained CNN model to predict class probabilities for the preprocessed image.
4. **Displaying Result**: 
   - Uses a threshold of 0.5 to determine whether the predicted probability indicates pneumonia or not.

## How to Run

1. Open the project folder in a command prompt.
2. Ensure you have the following dependencies installed:
   - Python
   - Streamlit
   - All necessary libraries (as listed in the requirements file or in the script)
3. Run the following command:
   ```
   streamlit run app.py
   ```
4. A localhost port will open in your browser.
5. Upload an image to classify, and the model will output the result ("Pneumonia/Normal") along with the confidence percentage.
