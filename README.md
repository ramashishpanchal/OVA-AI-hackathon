# 🛡️ AI-Powered Cyber Threat Intelligence Platform

> **Theme:** *Proactive Defense: AI-Driven Intelligence for the Frontlines of Cybersecurity*

## 📌 Overview
Modern cyber threats evolve faster than traditional security systems can handle. Security teams often face **huge volumes of network data**, making it challenging to quickly detect and classify attacks.

This project builds a **functional AI-powered Cyber Threat Intelligence platform** trained on **realistic network traffic data**. The platform predicts:
1️⃣ **Whether an attack occurred** (`label`)  
2️⃣ **The type of attack** (`attack_type`)  

This allows organizations to **proactively detect cyber threats** and prioritize mitigation efforts.

---

## 🎯 Objective
Build a **functional multi-output model** that:  
✅ Detects attacks in real-time (binary classification)  
✅ Classifies attacks into categories (multi-class classification)

---

## ⚡ Current Features
- **Dual Output Prediction:**  
  - `label` → Binary classification (Normal vs Attack)  
  - `attack_type` → Multi-class classification (DoS, Recon, Exploits, Fuzzers, etc.)  
- **Data-Driven Approach:** Trained from scratch on **UNSW-NB15 dataset**  
- **Interpretable Results:** Visualizations for analysts including **confusion matrices** and **feature importance**  
- **Functional Model:** Ready for offline testing and experimentation

---

## 🛠️ Planned / Future Features
- **Real-Time Inference:** Enable instant prediction for integration with **SOC dashboards or SIEM tools**  
- **Actionable Intelligence:** Context-aware recommendations to guide security teams in mitigating threats  
- **Automated Response:** Integration with playbooks and security orchestration tools  
- **Live Network Streaming:** Support for real-time network data monitoring

---

## 📂 Dataset
We use the **UNSW-NB15 dataset**:  
🔗 [UNSW-NB15 on Kaggle](https://www.kaggle.com/code/ramashishpanchal/hackathon)

- **Records:** 175,000 network flows  
- **Features:** 36 attributes (protocol, service, source/destination packets, TCP flags, etc.)  
- **Outputs:**  
  - `label` → Normal (0) / Attack (1)  
  - `attack_cat` → Attack category (Recon, Exploits, DoS, Fuzzers, etc.)

This dataset allows us to **train a functional model from scratch** that captures both **attack occurrence** and **attack type**.

---

## 📓 Kaggle Notebook
For a detailed walkthrough and interactive exploration of the model, check out our Kaggle notebook:
[AI-Powered Cyber Threat Intelligence Notebook](https://www.kaggle.com/your-username/your-notebook-name)

---

## 🏗️ Tech Stack
- **Backend:** Python  
- **ML/AI:** Scikit-learn, TensorFlow/Keras (Functional API)  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **App:** Streamlit (for interactive UI)

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-threat-intelligence-platform.git
cd ai-threat-intelligence-platform
