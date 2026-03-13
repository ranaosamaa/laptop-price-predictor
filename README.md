## Laptop Price Predictor

Predict the **estimated market price of a laptop** based on its hardware specifications using a **Lasso Regression** model wrapped in an interactive **Streamlit** web app.

The model is trained on a structured laptop dataset and exposes key configuration knobs such as RAM, CPU, storage, display quality (PPI), and more. The app returns an estimated price and a simple price range, along with a feature-importance view.

---

### Features

- **Interactive UI** built with `streamlit`
- **Lasso Regression** model with cross‑validated hyperparameter search
- **Feature importance** plot for the top predictors
- Handles **categorical features** via one‑hot encoding (company, type, CPU, GPU, OS)
- Simple **price range** estimate around the central prediction

---

### Project Structure

- `app.py` – Streamlit app for interactive price prediction and model explanation  
- `train_model.py` – Script to train and tune the Lasso Regression model and save the pipeline  
- `requirements.txt` – Python dependencies  
- `data/laptop_data.csv` – Input dataset used for training (not committed in some setups; see below)  
- `pipeline.pkl` – Trained `scikit-learn` pipeline (saved model used by the app)

---

### Installation

1. **Clone the repository**

```bash
git clone <YOUR_REPO_URL>.git
cd laptop-price-predictor
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
.\.venv\Scripts\activate  # on Windows
# source .venv/bin/activate  # on macOS / Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Dataset

The training script expects a CSV file at:

- `data/laptop_data.csv`

With at least:

- **Target column**: `Price (Euro)`  
- **Feature columns**: the engineered numeric and one‑hot encoded features used by the model.

If you are using the original raw laptop dataset, make sure to preprocess it into this feature format before running `train_model.py` (or extend the script accordingly).

> If you plan to publish this repository publicly and the dataset is proprietary, **do not commit** the CSV file. Instead, document where to obtain it.

---

### Training the Model

The model is trained using `train_model.py`, which:

- Loads `data/laptop_data.csv`
- Splits features `X` and target `y = "Price (Euro)"`  
- Builds a `Pipeline(StandardScaler() -> Lasso)`  
- Performs `RandomizedSearchCV` over the `lasso__alpha` hyperparameter using 5‑fold cross‑validation and R² scoring  
- Saves the best pipeline to `pipeline.pkl`

To (re)train the model:

```bash
python train_model.py
```

On success, you will see the best alpha and CV R² in the terminal, and `pipeline.pkl` will be created/updated in the project root.

---

### Running the Streamlit App

Ensure that:

- `pipeline.pkl` exists in the project root (run `python train_model.py` first if needed)

Then start the app with:

```bash
streamlit run app.py
```

By default, Streamlit will open the app in your browser at something like `http://localhost:8501`.

In the UI you can:

- Choose laptop **brand**, **type**, and **OS**
- Configure **RAM**, **CPU frequency**, **CPU family**, **GPU brand**
- Set **storage** (SSD/HDD), **touchscreen**, **IPS**, and **PPI**
- Click **“Predict Laptop Price”** to get:
  - Estimated price in Euro
  - Approximate 10% lower/upper bound range
  - A table summarizing your chosen configuration
  - A bar chart of top feature importances (when available)

---

### Model Details

- **Algorithm**: Lasso Regression (`sklearn.linear_model.Lasso`)  
- **Preprocessing**: `StandardScaler` on numeric features  
- **Hyperparameter search**: `RandomizedSearchCV` over `lasso__alpha` with a `loguniform(1e-5, 1e1)` prior  
- **Cross‑validation**: 5‑fold (`KFold`, shuffled, `random_state=42`)  
- **Metric**: R² (coefficient of determination)  

The Streamlit sidebar also shows the cross‑validated R² achieved by the tuned model.

---

### Requirements

Core dependencies (also listed in `requirements.txt`):

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

Install them with:

```bash
pip install -r requirements.txt
```

---

### Deployment Notes

- For deployment (e.g., on Streamlit Community Cloud or similar platforms), ensure that:
  - `requirements.txt` is up to date
  - `pipeline.pkl` is committed or generated as part of the deployment build step
  - The `data/laptop_data.csv` file is available at build time if you retrain during deployment

---

### License

Add your preferred license here (e.g., MIT, Apache 2.0). For example, you can include an `LICENSE` file and mention:

> This project is licensed under the MIT License – see the `LICENSE` file for details.

