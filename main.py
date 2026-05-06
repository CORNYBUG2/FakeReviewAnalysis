# =========================
# PyQt5 App using your saved files (same folder)
# =========================

# pip install pyqt5 transformers torch

import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QTextEdit, QPushButton, QLabel
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Load Model from CURRENT FOLDER
# =========================
MODEL_PATH = "."   # because model files are in same folder as main.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# =========================
# Prediction Function
# =========================
def predict_review(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    if pred == 1:
        return f"Fake Review ({confidence:.2f})"
    else:
        return f"Real Review ({confidence:.2f})"

# =========================
# UI
# =========================
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fake Review Detector")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Write review here...")
        layout.addWidget(self.text_edit)

        self.button = QPushButton("Predict")
        self.button.clicked.connect(self.run_prediction)
        layout.addWidget(self.button)

        self.result = QLabel("Result will appear here")
        layout.addWidget(self.result)

        self.setLayout(layout)

    def run_prediction(self):
        text = self.text_edit.toPlainText().strip()

        if not text:
            self.result.setText("Enter text")
            return

        output = predict_review(text)
        self.result.setText(output)

# =========================
# Run
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())