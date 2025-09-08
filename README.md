# Health and Housing Analysis Repository

This repository contains two primary projects: a **Heart Attack Prediction Model** and an analysis of the **S&P/Case-Shiller Home Price Index**. Below are the details for each project.

---

## 1. Heart Attack Prediction Model

### Introduction
This project aims to develop a predictive model for heart attack prediction using a dataset containing various health-related features. The goal is to maximize recall and precision metrics instead of accuracy, focusing on the importance of correctly identifying cases with a higher chance of heart attack.

### Features
The dataset includes the following features:
- **age**: Age of the patient
- **sex**: Sex of the patient
- **cp**: Chest pain type
  - 0 = typical angina
  - 1 = atypical angina
  - 2 = non-anginal pain
  - 3 = asymptomatic
- **trtbps**: Resting blood pressure in mm Hg
- **chol**: Cholesterol in mg/dl
- **exng**: Exercise induced angina
  - 1 = yes
  - 0 = no
- **fbs**: Fasting blood sugar > 120 mg/dl
  - 1 = true
  - 0 = false
- **restecg**: Resting electrocardiographic results
  - 0 = normal
  - 1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
  - 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria
- **thalachh**: Maximum heart rate achieved
- **slp**: Slope
- **caa**: Number of major vessels
- **thall**: Thalium stress test result
- **target**:
  - 0 = less chance of heart attack
  - 1 = more chance of heart attack

### Assignment Objectives
1. **Data Exploration and Analysis**:
   - Explore the dataset and analyze relationships between features.
   - Identify correlations and visualize feature distributions.
2. **Data Pre-processing**:
   - Handle missing values, outliers, and address unbalanced data.
   - Perform feature engineering, including handling correlated features and scaling.
3. **Model Building**:
   - Split the dataset into training and testing sets.
   - Choose appropriate models for heart attack prediction (e.g., Logistic Regression, Random Forest, XGBoost).
4. **Model Evaluation**:
   - Evaluate models using recall and precision metrics.
   - Visualize the confusion matrix for better understanding of model performance.
5. **Presentation**:
   - Create a non-code report using Jupyter Notebook or export as PDF.
   - Summarize findings, insights, and visualizations from the analysis.
6. **Explanation**:
   - Explain reasoning behind preprocessing steps, model selection, and metric choices.
   - Discuss the impact of certain features on heart attack prediction.

### Libraries/Package Used
1. `%matplotlib inline`: Enables inline plotting within a Jupyter Notebook environment.
2. `numpy`: Fundamental library for numerical computing in Python.
3. `pandas`: Powerful library for data manipulation and analysis.
4. `matplotlib.pyplot`: Comprehensive library for creating visualizations.
5. `seaborn`: High-level interface for statistical graphics.
6. `sklearn.model_selection`: Tools for splitting datasets into training and testing sets.
7. `collections`: Contains the Counter class for counting hashable objects.
8. `sklearn.linear_model`: Provides LogisticRegression.
9. `sklearn.metrics`: Offers metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
10. `sklearn.ensemble`: Contains RandomForestClassifier.
11. `xgboost`: Library for gradient boosted decision trees.
12. `warnings`: Manages warning messages.
13. `warnings.simplefilter("ignore")`: Suppresses warning messages (use with caution).

### Conclusion
- Logistic Regression emerges as the most suitable model for heart attack prediction, given its high recall and reasonable precision. Correctly identifying individuals with a higher chance of a heart attack is crucial for this task.
- Further optimization of hyperparameters and feature engineering may enhance the model's performance.
- Consider conducting additional analyses to understand feature impact on predictions and identify areas for improvement.

---

## 2. S&P/Case-Shiller Home Price Index Analysis

### 1. Objective
To analyze publicly available data on economic, demographic, and real estate indicators to build a predictive model that explains the impact of these factors on the S&P/Case-Shiller Home Price Index, a key indicator of U.S. home prices, over the last two decades.

### 2. Introduction
The S&P CoreLogic Case-Shiller Home Price Indices play a crucial role in tracking the price levels of single-family homes in the United States. The S&P CoreLogic Case-Shiller U.S. National Home Price Index provides a comprehensive view by aggregating data from nine regions and 20 major metropolitan areas, updated monthly. These indices measure percentage changes in housing market prices while maintaining a constant quality level, excluding variations due to house types, sizes, or physical characteristics.

### 3. Data and Methodology
#### 3.1 Data Collection
The features were identified by conducting a literature survey of The S&P CoreLogic Case-Shiller Home Price Indices. Most data was collected from [FRED](https://fred.stlouisfed.org/):
1. **UNRATE**: Unemployment Rate (Percent, Seasonally Adjusted, Monthly)
2. **CSUSHPISA**: S&P/Case-Shiller U.S. National Home Price Index (Index Jan 2000=100, Seasonally Adjusted, Monthly)
3. **PERMIT**: New Privately-Owned Housing Units Authorized: Total Units (Thousands of Units, Seasonally Adjusted Annual Rate, Monthly)
4. **PERMIT1**: New Privately-Owned Housing Units Authorized: Single-Family Units (Thousands of Units, Seasonally Adjusted Annual Rate, Monthly)
5. **MSACSR**: Monthly Supply of New Houses (Months' Supply, Seasonally Adjusted, Monthly)
6. **TTLCONS**: Total Construction Spending: Total Construction (Millions of Dollars, Seasonally Adjusted Annual Rate, Monthly)
7. **NASDAQCOM**: NASDAQ Composite Index (Index Feb 5, 1971=100, Not Seasonally Adjusted, Daily, Close)
8. **LFACTTTTUSM657S**: Active Population: Aged 15 and over: All Persons for United States (Growth rate previous period, Seasonally Adjusted, Monthly)
9. **HSN1F**: New One Family Houses Sold: United States (Thousands, Seasonally Adjusted Annual Rate, Monthly)
10. **HOUST1F**: New Privately-Owned Housing Units Started: Single-Family Units (Thousands of Units, Seasonally Adjusted Annual Rate, Monthly)
11. **LFPR**: Labor Force Participation Rate [](https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1966154)
12. **Housing Starts**: New Housing Project [](https://towardsdatascience.com/linear-regression-on-housing-csv-data-kaggle-10b0edc550ed)
13. **INDPRO**: Industrial Production: Cement [](https://fred.stlouisfed.org/series/INDPRO)
14. **Personal Income & Outlays**: CSV [](https://alfred.stlouisfed.org/release?rid=54)
15. **New Privately-Owned Housing Units Completed**: Total Units [](https://fred.stlouisfed.org/series/computsa)

#### 3.2 Data Preparation
When combining datasets with monthly and quarterly data, missing values are replaced with the mode value to ensure completeness and consistency.

#### 3.3 Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed using ECDF (Empirical Cumulative Distribution Function) for effective histogram visualization and regression plotting to examine dataset variations.

#### 3.4 Model Selection and Evaluation
Models evaluated include linear regression, random forest, and XGBoost. The random forest model performed best with an MSE of 8.33 and an adjusted R² of 0.99.

### 4. Results and Discussion
#### 4.1 Exploratory Data Analysis
HNFSEPUSSA follows the same trend as CSUSHPISA. A small change in NA000334Q can significantly impact the S&P Home Price Index (HPI), with a positive correlation. LFPR and TTLCONS are positively correlated with HPI. UNRATE is inversely correlated with employment.

#### 4.2 Correlation Matrix
- New Privately-Owned Housing Units Started: Single-Family Units (0.94 with CSUSHPISA)
- Total Construction Spending (TTLCONS) (0.4 with CSUSHPISA)
- Monthly Supply of New Houses (0.84 with CSUSHPISA)
- New One Family Houses Sold (0.55 with CSUSHPISA)
- NASDAQ Composite Index (0.26 with CSUSHPISA)

#### 4.3 Machine Learning Models
Random forest was chosen for its balance of low MSE (8.33) and high R² (0.997), effectively identifying important parameters.

### 5. Conclusions
- Key influencing factors on the S&P Home Price Index (HPI) include:
  - **Personal Income & Outlays**: Directly reflects changes in home prices.
  - **Employment rate**: Increases demand for housing and home prices.
  - **Total Construction Spending (TTLCONS)**: Affects housing supply and demand.
  - **New privately owned housing**: Directly measures property prices.
  - **CPI-Adjusted Price**: Reflects changes in housing costs.
  - **NASDAQ Composite Index (NASDAQCOM)**: Influences economic growth and housing demand.
  - **Monthly Supply of New Houses (MSACSR)**: Impacts home prices based on supply relative to demand.

### MECE Framework
#### Factors Influencing US House Prices
- **Economic Factor**:
  - Growth in the Economy
  - Unemployment
  - Customer Trust Rates
  - Offering
  - GDP
  - Home Sales Economy Mirror
  - Supply and Demand
  - Advance
  - Compliance
  - Competition
  - Extinct
  - Surplus Productivity
- **Location**:
  - Neighborhoods
  - Highways
  - Attractions
  - Schools
  - Area Desirability
  - Crime Rate
- **Government**:
  - Government laws
  - Property Taxes
- **Banks**:
  - Mortgage Availability
  - Interest Rates

---

## Usage
- Clone the repository and use the provided data sources.
- Scripts for data processing and modeling are included for both projects.

## License
[MIT License](https://opensource.org/licenses/MIT)
