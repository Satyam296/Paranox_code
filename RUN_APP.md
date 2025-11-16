# 🚀 Running the Space Station Safety Detection App

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_app.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Use the App
1. The app will open in your browser (usually `http://localhost:8501`)
2. Upload a space station image using the file uploader
3. Adjust the confidence threshold in the sidebar if needed
4. View detection results with bounding boxes
5. Check the detection summary for statistics

## Features

✅ **Real-time Detection**: Upload and detect instantly  
✅ **Interactive UI**: Adjustable confidence threshold  
✅ **Visual Results**: Side-by-side comparison of original and detected images  
✅ **Detailed Stats**: Object counts, confidence scores, and class breakdown  
✅ **7 Safety Objects**: Oxygen Tank, Nitrogen Tank, First Aid Box, Safety Switch Panel, Fire Extinguisher, Fire Alarm, Emergency Phone

## Model Performance

- **mAP@0.5**: 54.67%
- **Precision**: 82.26%
- **Recall**: 47.16%
- **Inference Speed**: ~48.5ms per image

## Troubleshooting

**Model not found error?**
- Ensure `scripts/runs/train/exp4/weights/best.pt` exists
- Run training first if model is missing

**Import errors?**
- Install all requirements: `pip install -r requirements_app.txt`

**Port already in use?**
- Run with different port: `streamlit run app.py --server.port 8502`

## Demo Video Recording Tips

1. Start the app
2. Show the main interface
3. Upload a test image
4. Demonstrate detection results
5. Show confidence adjustment
6. Display detection statistics

Record 30-60 seconds showing these features!
