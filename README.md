# Indian Vehicle Detection — Streamlit

A Streamlit image-upload interface for running inference with trained Ultralytics YOLO vehicle-detection weights.

## Files

```text
vehicle-detection-app/
├── app.py
├── model.pt
├── requirements.txt
└── README.md
```

Copy your trained weights into this folder and name them `model.pt`. The app uses the class names stored in the model itself, so a separate dataset YAML or Roboflow API key is not needed for inference.

## Run locally

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Notes

- This app accepts image files only; it does not process video.
- Lower confidence thresholds can increase detections and false positives.
- Larger inference sizes can help with small objects but use more memory and processing time.
- Keep API keys and other secrets out of source code and version control.
