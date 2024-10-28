# Gujarat Solar Power Analyzer

# WARNING, very WIP - code is public despite not being meant for public

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Visualizations](#visualizations)
- [Logging](#logging)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The **Gujarat Solar Power Analyzer** is a Python-based application designed to:

- **Fetch** solar and meteorological data for key cities in Gujarat from the NASA POWER API.
- **Process** and **clean** the data to ensure accuracy and reliability.
- **Train** a machine learning model to predict solar power potential based on various weather parameters.
- **Visualize** the analyzed data and predictions through interactive plots.

This tool assists in understanding solar power potential across different regions of Gujarat, aiding in strategic planning and sustainable energy initiatives.

## Features

- **Data Retrieval**: Fetches daily solar and meteorological data for multiple cities.
- **Data Cleaning**: Ensures data integrity by handling missing values and outliers.
- **Solar Power Calculation**: Computes theoretical solar power output based on irradiance and environmental factors.
- **Machine Learning**: Trains a Random Forest Regressor to predict solar power from weather parameters.
- **Interactive Visualizations**: Generates insightful plots using Plotly, including:
  - Regional power potential comparisons.
  - Monthly trend analyses.
  - Geographic distribution maps.
  - Weather impact scatter plots.
- **Logging**: Comprehensive logging for monitoring data processing and model training stages.
- **Data Export**: Saves processed data to a CSV file for further analysis or record-keeping.

## Getting Started

Follow these instructions to set up and run the Gujarat Solar Power Analyzer on your local machine.

### Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from [here](https://www.python.org/downloads/).

- **pip**: Python package installer. It usually comes bundled with Python.

### Program Workflow
Data Fetching: Retrieves solar and weather data for the past 365 days for selected cities in Gujarat.
Data Processing: Cleans and preprocesses the data.
Model Training: Trains a Random Forest model to predict solar power potential.
Visualization: Generates and displays interactive plots.
Data Export: Saves the processed data to gujarat_solar_data.csv.
Output
Visualizations: Four interactive Plotly plots will open in your default web browser.
CSV File: A gujarat_solar_data.csv file containing the processed data will be saved in the project directory.
Logs: Console logs provide insights into the program's execution and any warnings or errors encountered.
Data Processing
The application processes the following parameters from the NASA POWER API:

ALLSKY_SFC_SW_DWN: Solar radiation on a horizontal surface (W/m²)
T2M: Temperature at 2 meters (°C)
RH2M: Relative humidity at 2 meters (%)
CLOUD_AMT: Cloud amount (oktas)
ALLSKY_KT: Clearness index (unitless)
WS2M: Wind speed at 2 meters (m/s)
Solar Power Calculation
Theoretical solar power is calculated using the formula:

[
\text{Power (kW)} = \left(\frac{\text{Solar Radiation (W/m²)} \times 24}{1000}\right) \times \left(1 - 0.05 \times \text{Cloud Cover}\right) \times \text{Clearness Index} \times \left(1 - 0.002 \times (\text{Temperature} - 25)\right) \times \text{Panel Efficiency} \times \text{Area} / 24
]

Panel Efficiency: 18%
Solar Panel Area: 1000 m²
Note: The formula ensures proper unit conversion and accounts for environmental factors affecting solar power output.

Model Training
A Random Forest Regressor is employed to predict solar power potential based on the following features:

Temperature
Humidity
Cloud Cover
Clearness Index
Wind Speed
Training Process
Data Splitting: The dataset is divided into training (80%) and testing (20%) subsets.
Scaling: Features are standardized using StandardScaler.
Model Training: The Random Forest model is trained on the scaled training data.
Evaluation: Model performance is assessed using Mean Squared Error (MSE) on the test set.
Evaluation
The program logs the Mean Squared Error (MSE) to provide insights into the model's predictive accuracy.

Visualizations
Interactive visualizations are created using Plotly to provide a comprehensive analysis of solar power potential.

Regional Comparison
Type: Box Plot
Description: Compares predicted solar power across different regions in Gujarat.
Axes:
X: Region
Y: Predicted Solar Power (kW)
Monthly Trends
Type: Line Chart
Description: Shows the monthly average predicted solar power generation for each city.
Axes:
X: Month
Y: Average Predicted Solar Power (kW)
Geographic Distribution
Type: Scatter Mapbox
Description: Displays the geographic distribution of predicted solar power potential across Gujarat cities.
Features:
Size of markers represents solar power potential.
Colors differentiate regions.
Weather Impact Analysis
Type: Scatter Plot
Description: Analyzes the relationship between temperature and predicted solar power, colored by clearness index.
Axes:
X: Temperature (°C)
Y: Predicted Solar Power (kW)
Color: Clearness Index
All visualizations are interactive and can be explored in a web browser.

Logging
Comprehensive logging is implemented to monitor the program's execution and facilitate debugging.

Log Levels:
DEBUG: Detailed information for diagnosing issues.
INFO: General information about program progress.
WARNING: Alerts about potential issues (e.g., missing data).
ERROR: Critical problems preventing normal operation.
Log Output: Logs are printed to the console with timestamps and severity levels.
Example Log Entry:


2023-10-10 12:00:00 - __main__ - INFO - Data after cleaning: 3650 records
Project Structure

gujarat-solar-analyzer/
├── solar_analyzer.ipynb
├── gujarat_solar_data.csv
├── README.md
└── LICENSE

Contributing
Contributions are welcome! Please follow these steps:

Fork the Repository
Create a Feature Branch

git checkout -b feature/YourFeature
Commit Your Changes

git commit -m "Add Your Feature"
Push to the Branch

git push origin feature/YourFeature
Open a Pull Request
Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation and testing.

License
This project is licensed under the MIT License.

Acknowledgements
NASA POWER API: Providing comprehensive solar and meteorological data.
Pandas & NumPy: Essential libraries for data manipulation and analysis.
Scikit-learn: Powerful tools for machine learning.
Plotly: Creating interactive and informative visualizations.
Loguru Logging: Simplifying logging processes.
Developed with ❤️ by Abhi.
