# WaveFilter: Advanced Imaging for Marine Microplastic Monitoring

![Macroplastic](https://github.com/user-attachments/assets/3255a5ee-829c-4f49-928d-bccca004f2eb)

---

## Overview
WaveFilter is a state-of-the-art microplastic detection system that leverages the YOLOv5 deep learning model for identifying and monitoring microplastics in marine environments. The project aims to provide an automated and scalable solution for tackling the growing challenge of marine microplastic pollution, combining cutting-edge machine learning with user-friendly web interfaces.

---

## Features
- **Efficient Detection**: Achieves 79.8% precision and 67.1% recall.
- **Compact and Deployable**: YOLOv5 with 214 layers and a size of 14.4 MB.
- **Web-based Interface**: Built with Flask, enabling users to upload images for real-time detection.
- **Advanced Image Processing**: Handles complex marine backgrounds with high reliability.

---

## Motivation
Microplastics pose a severe threat to marine ecosystems and human health. Current detection methods are labor-intensive and lack scalability. WaveFilter offers an innovative approach to automate and accelerate the detection process, aiding researchers and policymakers in addressing this environmental crisis.

---

## Methodology
1. **Data Collection and Annotation**: 
   - High-resolution images from public repositories like Kaggle.
   - Annotation using tools such as LabelImg.

2. **Model Training**: 
   - Fine-tuned YOLOv5 using transfer learning.
   - Achieved an mAP of 72.1% at IoU=0.5.

3. **Deployment**: 
   - Flask-based server for backend processing.
   - Vercel for hosting the web interface.

---

## Project Structure
```
WaveFilter-Advanced-Imaging-for-Marine-Microplastic-Monitoring/
│
├── api/                     # Backend APIs
├── results/                 # Model output results
├── templates/               # HTML files for the web interface
├── uploads/                 # Uploaded user images
├── .gitignore               # Git ignore file
├── best_microplastic_model.pt # Trained YOLOv5 model
├── requirements.txt         # Dependencies
├── vercel.json              # Deployment configuration for Vercel
├── wsgi.py                  # WSGI configuration
└── README.md                # Project documentation
```

---

## Requirements
Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ishan22399/WaveFilter-Advanced-Imaging-for-Marine-Microplastic-Monitoring.git
   cd WaveFilter-Advanced-Imaging-for-Marine-Microplastic-Monitoring
   ```
2. Run the Flask server:
   ```bash
   python wsgi.py
   ```
3. Access the web interface at `http://localhost:5000`.

---

## Results
- **Model Performance**: 
  - Precision: 79.8%
  - Recall: 67.1%
  - mAP (IoU=0.5): 72.1%
- **Visual Outputs**: Bounding boxes accurately identify microplastics in images, demonstrating robustness in real-world conditions.

---

## Future Work
- Extend detection capabilities to include larger debris.
- Optimize model for underwater drone integration.
- Enhance scalability for real-time environmental monitoring.

---

## Contribution
    Contributions are welcome! Please open an issue or submit a pull request.
---
