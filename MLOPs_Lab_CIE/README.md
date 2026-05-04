# 🛠️ PipeWatch Corrosion Risk Prediction (MLOps Lab CIE)

This repository contains a complete MLOps pipeline for predicting corrosion risk scores in pipeline infrastructure.

## 📁 Repository Structure

Internals_Basics/
 └── MLOPs_Lab_CIE/
      ├── data/
      ├── src/
      ├── models/
      ├── results/
      ├── requirements.txt
      └── .gitignore

## ⚙️ Setup

python -m venv .venv
.venv\Scripts\activate
pip install -r MLOPs_Lab_CIE/requirements.txt

## 🚀 Run

python MLOPs_Lab_CIE/src/train.py
python MLOPs_Lab_CIE/src/tune.py
python MLOPs_Lab_CIE/src/api.py
python MLOPs_Lab_CIE/src/register_model.py

## 📊 Outputs

- results/step1_s1.json
- results/step2_s2.json
- results/step3_s4.json
- results/step4_s6.json

## 🌐 API

GET /status  
POST /forecast

## 👨‍💻 Author

Saikumar R
