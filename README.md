# FlyABC Flight Delay Prediction Project

## 1. Business Context & Problem Statement

FlyABC Airline measures its operational performance primarily through On-Time Performance (OTP) of departure flights. A flight is considered delayed if it departs more than 15 minutes after its scheduled time. Improving OTP is crucial for reducing operational costs, enhancing customer satisfaction, and maintaining regulatory compliance.

This project addresses a data science case study provided by FlyABC.

## 2. Objective

The primary objective of this project is to develop a predictive machine learning model that estimates the probability of a flight departure delay (more than 15 minutes) for upcoming flights. This model aims to enable FlyABC's operations and ground staff to proactively manage resources, prioritize high-risk flights, and mitigate potential delays.

## 3. Data

The project utilizes a historical dataset of FlyABC's flight operations. The sample data (`Flight_Delay_Case_Sample_Data (1).xlsx - Sheet1.csv`) includes features such as:

* `Flight_ID`
* `departure date`
* `parking_bay`
* `sch_departure_lt` (Scheduled Departure Local Time)
* `sch_departure_utc` (Scheduled Departure UTC)
* `from_airport`
* `to_airport`
* `to_Region`
* `terminal`
* `gate_number`
* `flight_minutes`
* `booked_passengers`
* `flt_capacity`
* `prev_Airport` (Previous Airport)
* `prev_flight_arrival` (Previous Flight Arrival Time)
* `total_number_of_bags`
* `distance`
* `delay` (Actual delay in minutes, used to derive the target variable)
* `aircraft`

The target variable for prediction is `is_delayed` (binary: 1 if delay > 15 minutes, 0 otherwise).

## 4. Project Structure & Workflow

This project is structured  a Jupyter Notebook that perform the following key steps:

### 4.1. Initial Data Loading and EDA (Conceptual - part of initial analysis)

* Load the raw flight data.
* Perform initial Exploratory Data Analysis (EDA) to understand data types, missing values, distributions, and basic statistics.
* Create the binary target variable `is_delayed` from the `delay` column.

### 4.2. Feature Engineering 
* **Date/Time Features**:
    * Convert date/time strings to datetime objects.
    * Extract features like hour, day of the week, month, day of the year
    * Calculate `turnaround_time_minutes` (time between previous flight arrival and current scheduled departure).
* **Load Factor**: Calculate `load_factor` as `booked_passengers / flt_capacity`.
* **Categorical Feature Handling**:
    * Drop single-value columns (e.g., `from_airport`).
    * One-hot encode low-cardinality categorical features (e.g., `to_Region`, `terminal`, `aircraft`).
    * Convert high-cardinality categorical features to string type for initial handling.
* Drop redundant or original columns used for feature creation.
* Output: `flights_feature_engineered.csv` (data ready for modeling).

### 4.3. Model Building & Evaluation (`baseline_model_script_v1.ipynb` or similar)

This script focuses on building, training, and evaluating predictive models.

* **Data Preparation**:
    * Load the feature-engineered data.
    * Drop high-cardinality string columns (e.g., `Flight_ID`, `parking_bay`) for baseline models.
    * Handle any remaining NaN/infinite values by imputation (median).
    * Split data into training and testing sets (stratified by the target variable).
    * Scale numerical features using `StandardScaler`.
* **Model 1: Random Forest Classifier (Baseline)**
    * Train a `RandomForestClassifier` with `class_weight='balanced'` to handle initial class imbalance.
    * Evaluate on training and testing sets using accuracy, ROC AUC, classification report (precision, recall, F1-score), and confusion matrix.
    * Analyze feature importances.
    * *Observation*: This model showed signs of overfitting (perfect training scores, lower test scores).
* **Model 2: XGBoost Classifier with SMOTE (Improved Approach)**
    * Apply SMOTE (Synthetic Minority Over-sampling Technique) to the *training data only* to address class imbalance more directly.
    * Train an `XGBClassifier`.
    * Evaluate on the (SMOTE'd) training data and the original (unseen) test data.
    * Analyze feature importances.
* **Evaluation Metrics**:
    * Accuracy
    * ROC AUC Score
    * Classification Report (Precision, Recall, F1-Score)
    * Confusion Matrix

## 5. How to Run

1.  **Prerequisites**:
    * Python 3.x
    * Required libraries: `pandas`, `numpy`, `scikit-learn`, `imblearn` (for SMOTE), `xgboost`, `matplotlib`, `seaborn`.
    * You can install them using pip:
        ```bash
        pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
        ```
2.  **Data**:
    * Ensure the raw data file (`Flight_Delay_Case_Sample_Data (1).xlsx - Sheet1.csv`) is available.
    * The scripts will generate intermediate files (`flights_with_target.csv`, `flights_feature_engineered.csv`).
3.  **Execution**:
    * Run the initial data processing/target creation step (if separated).
    * Run the feature engineering script (e.g., `python feature_engineering_script.py` or execute cells in a notebook).
    * Run the modeling script (e.g., `python modeling_script.py` or execute cells in a notebook). The script currently contains logic for both Random Forest and XGBoost with SMOTE; you might run them sequentially or comment out parts to test individually.

## 6. Key Findings & Potential Next Steps

* **Feature Importance**: Features related to `terminal`, `departure_day_of_week`, `load_factor`, and `turnaround_time_minutes` appear to be influential.
* **Class Imbalance**: The dataset has a notable class imbalance (more non-delayed flights). Techniques like SMOTE or using `class_weight`/`scale_pos_weight` are important.
* **Overfitting**: Initial models (like an unconstrained Random Forest) can overfit. Hyperparameter tuning is crucial.
* **Model Performance**: XGBoost with SMOTE provides a more robust approach. Further tuning and feature engineering could improve performance.

**Next Steps Could Include**:

* **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for both Random Forest and XGBoost.
* **Advanced Feature Engineering**:
    * Better encoding for high-cardinality categorical features (e.g., target encoding, embedding layers).
    * Interaction features.
    * Cyclical encoding for time-based features.
* **External Data**: Incorporate weather data, as discussed in the case study, which could significantly impact delay prediction
* **Alternative Models**: Explore other algorithms (e.g., LightGBM, CatBoost, Neural Networks).

## 7. Case Study Deliverables

This project work forms the basis for a presentation covering:
* Problem Understanding and Business Impact
* Data Exploration and Analysis (including additional data sources like weather)
* Feature Engineering
* Model Selection and Machine Learning Approach
* Model Validation and Evaluation
* Deployment and Monitoring Plan
* Communication and Business Recommendations


