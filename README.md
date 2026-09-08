# Handwritten Digit Recognition

A deep learning project for recognizing handwritten digits from the MNIST dataset. It features a custom Convolutional Neural Network (CNN) implemented from scratch using NumPy, baseline and data-augmented Keras models, and an interactive desktop application built with Tkinter.

## Built With

- **Python 3**: Core programming language.
- **NumPy**: Custom CNN implementation built from scratch.
- **TensorFlow / Keras**: Baseline and data-augmented deep learning models.
- **Tkinter & Pillow**: Desktop GUI interface and canvas image processing.
- **SciPy**: Center-of-mass image normalization and pre-processing.
- **Pandas, Scikit-learn, Matplotlib, Seaborn**: Evaluation metrics and data visualization.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bartek03m/digit-recognition.git
   cd digit-recognition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the interactive GUI:
```bash
python gui/gui.py
```

Draw a digit on the canvas using your mouse, choose a model (Custom NumPy CNN, Baseline Keras, or Augmented Keras), and click **Predict** to view classification results and confidence probabilities.

## Model Results

| Model | Training Samples | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Keras CNN (Baseline) | 60,000 | 98.80% | 0.99 | 0.99 | 0.99 |
| Keras CNN (Augmented) | 1,080,000 | 99.22% | 0.99 | 0.99 | 0.99 |
| Custom CNN (NumPy) | 180,000 | 99.15% | 0.99 | 0.99 | 0.99 |


## License

Distributed under the MIT License. See `LICENSE` for more information.
