## Installation

1. Clone the repository:
   $ git clone <repository-url>
   $ cd <project-directory>
2. Create and activate a virtual environment:
   $ python -m venv venv
   $ source venv/bin/activate
3. Install dependencies:
   $ pip install -r requirements.txt

## Usage

1. Download the Dataset:
   $ python download_dataset.py
   $ cp data/download/adult.csv data/raw/adult.csv
2. Preprocess the Data:
   $ python process_dataset.py
3. Train the Models
   $ python train.py
4. (Optional) Generate SHAP Interpretations
   $ python shap_interpretation.py
5. Run the Prediction API
   $ python app.py

## Making predictions

1. Send a POST request to /predict with JSON data:
   $ curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @for_test.json
