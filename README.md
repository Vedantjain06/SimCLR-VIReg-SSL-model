
# Time Series Classification Using Self-Supervised Learning

This project aims to improve time series classification accuracy by leveraging self-supervised learning (SSL) techniques. Using the UWaveGestureLibrary dataset as the primary test dataset, we trained models using pretext tasks on a separate dataset to learn feature representations and fine-tuned them for gesture recognition.

## Project Overview

The project uses self-supervised learning methods, specifically VICReg and SimCLR, to extract meaningful features from unlabeled time series data. These features are then used to train a classifier that achieves higher accuracy in time series gesture recognition. We perform the following steps:

1. **Dataset Preparation**: Clean and preprocess the UWaveGestureLibrary dataset and other datasets used for pre-training.
2. **Self-Supervised Learning Implementation**: Train feature extraction models using VICReg and SimCLR on an alternative dataset (e.g., HAR dataset) to learn representations.
3. **Classification Model**: Fine-tune a classifier on the UWaveGestureLibrary dataset using the learned representations.
4. **Performance Verification**: Evaluate model accuracy and improvements gained from the self-supervised approach.

## Prerequisites

- Python 3.x
- Required libraries: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`
- Recommended to use a virtual environment to manage dependencies

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/yourprojectname.git
    cd yourprojectname
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Dataset Preparation

   - Download the UWaveGestureLibrary dataset [here](https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary) and place it in the `data/` folder.
   - Preprocess the data by running:
     ```bash
     python preprocess_data.py
     ```
   - This will clean, scale, and handle missing values in the dataset.

### 2. Self-Supervised Learning

   - Train the self-supervised models (VICReg and SimCLR) on the pre-training dataset (e.g., HAR dataset) by running:
     ```bash
     python ssl_training.py
     ```
   - Adjust hyperparameters and dataset paths as needed in the `ssl_training.py` script.

### 3. Fine-Tuning and Classification

   - Fine-tune a classification model using the representations learned from self-supervised learning:
     ```bash
     python classifier_training.py
     ```

### 4. Performance Verification

   - Evaluate the model’s accuracy and compare it with baseline methods by running:
     ```bash
     python evaluate.py
     ```

### 5. Streamlit Dashboard (Optional)

   - To visualize the results in a web app, run the Streamlit dashboard:
     ```bash
     streamlit run app.py
     ```

## Directory Structure

```
.
├── data/                       # Data directory
├── src/                        # Source code
│   ├── preprocess_data.py      # Data preprocessing script
│   ├── ssl_training.py         # Self-supervised learning script
│   ├── classifier_training.py  # Classification model training script
│   ├── evaluate.py             # Evaluation script
├── app.py                      # Streamlit app for visualization
├── requirements.txt            # Required libraries
└── README.md                   # Project overview and instructions
```

## Results

The project demonstrates that self-supervised learning techniques like VICReg and SimCLR can improve classification accuracy on the UWaveGestureLibrary dataset by leveraging feature representations learned from other time series datasets.

## Future Work

Future work could explore GAN-based data augmentation and CLIP-inspired multimodal learning to enhance the model's generalization abilities on time series data.

## License

This project is licensed under the MIT License.

---

This README provides a concise guide on how to set up, run, and understand the project. Adjust specific paths, descriptions, and configurations as needed based on your project's final structure.
