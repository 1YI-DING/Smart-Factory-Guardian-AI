# Smart-Factory-Guardian-AI
Smart Factory Guardian: An AI-driven platform for industrial equipment failure early-warning and life-cycle management.

# 🏭 Predictive Maintenance for Smart Manufacturing

## 📖 About the Project
This project provides an end-to-end AI solution for monitoring industrial equipment health and predicting potential failures before they occur. Unlike static threshold-based systems, this project implements a **physical degradation model** to simulate realistic sensor data (Vibration, Temperature, Pressure) as equipment ages.

The system integrates a complete machine learning pipeline—from data synthesis and feature engineering to model serving and real-time visualization.

### 🌟 Key Features
* **Realistic Data Synthesis**: Generated IoT sensor data based on industrial degradation trends (linear/non-linear wear-and-tear).
* **Dual-Model Strategy**: 
    * **Classification**: Random Forest to predict failure probability.
    * **Anomaly Detection**: Isolation Forest to identify out-of-distribution operating patterns.
* **Microservice Architecture**: High-performance API powered by **FastAPI** for real-time model inference.
* **Interactive Dashboard**: A professional digital-twin monitoring interface built with **Streamlit** and **Plotly**.
* **RUL Awareness**: Incorporates Remaining Useful Life (RUL) metrics for advanced maintenance scheduling.

### 🛠️ Tech Stack
* **Language**: Python 3.9+
* **Data Science**: Pandas, NumPy, Scikit-learn
* **Web Frameworks**: FastAPI (Backend), Streamlit (Frontend)
* **Visualization**: Plotly, Seaborn
* **Environment**: Conda / Pip / Docker (Optional)

# 🚀 Quick Start

Follow these steps to get the system up and running on your local machine.

### 1. Prerequisites
* **Python 3.9** or higher
* **Conda** or **venv** (recommended)

### 2. Installation & Setup
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/yiiiiiiding/Smart-Factory-Guardian-AI.git](https://github.com/yiiiiiiding/Smart-Factory-Guardian-AI.git)
cd Smart-Factory-Guardian-AI

# Install dependencies
pip install -r requirements.txt
```
### 3. Running the Pipeline
The project follows a linear workflow. Please run the scripts in the following order:
* **Step 1:** Generate Synthetic Data
```bash
python generate_realistic_data.py
```
* **Step 2:** Train the Models

Train the Random Forest and Isolation Forest models and save them as .pkl files.
```bash
python train.py
```
* **Step 3:** Start the Backend API (FastAPI)

Launch the inference service.
```bash
python api.py
```
Once started, visit http://127.0.0.1:8000/docs to test the API via Swagger UI.
* **Step 4:** Launch the Dashboard (Streamlit)

Open a new terminal and run the interactive monitoring panel.
```bash
python -m streamlit run app.py
```
The dashboard will automatically open in your default browser at http://localhost:8501.

# 💡 Notes for Beginners
* **Model Files:** Ensure rf_model.pkl, iso_model.pkl, and feature_names.pkl are generated in the root directory before starting the API or Dashboard.
* **Environment:** If you are using PyCharm, make sure your Terminal is switched to the correct Conda environment.
* **Ports:** Ensure ports 8000 (FastAPI) and 8501 (Streamlit) are not being used by other applications.
