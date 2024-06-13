# Data Science Project: VM Performance and Resource Allocation Optimization

## Description
This project aims to optimize virtual machine (VM) performance and resource allocation by analyzing various metrics related to virtualization, system performance, and resource usage. The goal is to provide insights and recommendations that improve VM performance, resource efficiency, and overall system utilization. This involves preprocessing data from multiple sources, conducting exploratory data analysis (EDA), feature engineering, training predictive models, and presenting the findings through visualizations and reports.

### Objectives
- **Optimize VM Performance:** Enhance the efficiency and performance of VMs by analyzing relevant metrics.
- **Improve Resource Allocation:** Identify opportunities for better resource utilization and allocation.
- **Provide Actionable Insights:** Generate insights and recommendations to guide resource optimization strategies.

### Key Metrics
- **VM Metrics:** CPU utilization, memory usage, disk I/O operations.
- **OpenShift Metrics:** Container resource usage, pod performance, cluster resource utilization.
- **RHEL Metrics:** System load average, network I/O throughput, context switching rate.
- **Insights Metrics:** Resource bottlenecks, capacity planning recommendations, anomaly detection alerts.

### Expected Outcomes
- Reduction in VM resource contention.
- Improved VM performance and efficiency.
- Increased accuracy in resource allocation.
- Comprehensive visualizations and reports demonstrating business metrics.

## Task List for Achieving Predictive Model and Business Metrics

### 1. Data Preprocessing
- [ ] **Load and Rename Datasets:**
  - Ensure all datasets are loaded and columns are renamed as per the mappings.
- [x] **Combine Datasets:**
  - Concatenate all the datasets to form a single combined dataframe.
- [x] **Encode Non-Numeric Columns:**
  - Apply label encoding to categorical features.

### 2. Exploratory Data Analysis (EDA)
- [x] **Plot Original Correlation Matrix:**
  - Visualize the correlation between different metrics.
- [x] **Plot Filtered Correlation Matrix:**
  - Highlight significant correlations for further analysis.
- [x] **Plot Distributions:**
  - Visualize the distributions of key metrics and annotate insights.

### 3. Feature Engineering
- [ ] **Create New Features:**
  - Develop new features based on domain knowledge and correlation analysis.
  - Example: Resource utilization ratios, peak usage times, anomaly scores.
- [x] **Normalize Data:**
  - Normalize the data using StandardScaler.

### 4. Model Training and Evaluation
- [ ] **Train Predictive Models:**
  - Implement and train models such as Random Forest, ARIMA for time series analysis.
- [ ] **Evaluate Model Performance:**
  - Calculate and log performance metrics such as MAE, RMSE.
- [ ] **Tune Hyperparameters:**
  - Perform hyperparameter tuning to optimize model performance.

### 5. Insights and Recommendations
- [ ] **Resource Utilization Analysis:**
  - Analyze and visualize resource usage to identify bottlenecks.
- [ ] **Clustering Analysis (K-Means):**
  - Group VMs based on resource usage patterns to identify under or overutilized VMs.
- [ ] **Time Series Analysis (ARIMA):**
  - Forecast future resource usage trends to optimize allocation.

### 6. Business Metrics and Visualizations
- [ ] **KPIs Calculation:**
  - Calculate key performance indicators such as reduction in resource contention, improved VM performance.
- [ ] **Visualize Business Metrics:**
  - Create visualizations to demonstrate business metrics and insights.
  - Example: Line graphs for resource usage over time, cluster plots for VM groups.

### 7. Documentation and Reporting
- [ ] **Document Analysis and Insights:**
  - Prepare detailed documentation of the data analysis process, insights, and recommendations.
- [ ] **Generate Reports:**
  - Compile the findings into a comprehensive report for stakeholders.
- [ ] **Create Presentation:**
  - Develop a presentation to communicate the results and business impact.

## Detailed Task List with Subtasks

### 1. Data Preprocessing
- Load and rename datasets (kubevirt, openshift, insights, rhel).
- Combine datasets into a single dataframe.
- Encode non-numeric columns (Cluster Node Health, Network Traffic Patterns).

### 2. Exploratory Data Analysis (EDA)
- Plot and analyze the original correlation matrix.
- Plot and highlight significant correlations in the filtered correlation matrix.
- Plot distributions of key metrics and annotate insights.

### 3. Feature Engineering
- Create new features based on analysis (e.g., resource utilization ratios).
- Normalize the data using StandardScaler.

### 4. Model Training and Evaluation
- Train predictive models (Random Forest, ARIMA).
- Evaluate models and log performance metrics (MAE, RMSE).
- Tune model hyperparameters.

### 5. Insights and Recommendations
- Perform resource utilization analysis.
- Conduct clustering analysis to group VMs.
- Execute time series analysis for forecasting.

### 6. Business Metrics and Visualizations
- Calculate key performance indicators (KPIs).
- Visualize business metrics (line graphs, cluster plots).

### 7. Documentation and Reporting
- Document analysis process and insights.
- Generate comprehensive reports for stakeholders.
- Create a presentation to communicate results and impact.


Resource Bottleneck Prediction:

Objective: Predict potential resource bottlenecks based on current and historical usage patterns.
Value: Helps in addressing bottlenecks before they impact performance, improving overall system efficiency.

System Load Forecasting:

Objective: Forecast future system load to optimize scheduling and resource allocation.
Value: Enables efficient use of resources, improving system performance and reducing latency.

CPU Usage Prediction:

Objective: Predict future CPU usage based on historical data to ensure optimal resource allocation and avoid over-provisioning.
Value: Helps in planning and scaling infrastructure to meet demand, reducing costs, and improving performance.


Network I/O Prediction:

Objective: Predict future network I/O throughput to ensure sufficient bandwidth and avoid network congestion.
Value: Ensures smooth and uninterrupted data flow, enhancing user experience and service reliability.


## Task List for VM Performance and Resource Allocation Optimization





3. **Feature Engineering**
   - Create new features that might improve model accuracy, such as the ratio of CPU to memory utilization.
   - Select relevant features for modeling.

4. **Data Splitting**
   - Split the data into training and testing sets using `train_test_split`.

5. **Handling Imbalanced Data**
   - Address any class imbalances using techniques like SMOTE.

6. **Model Training and Tuning**
   - Train a Random Forest model to predict the risk score.
   - Fine-tune the model parameters using `RandomizedSearchCV`.
   - Validate the model using cross-validation techniques to ensure robustness.

7. **Model Evaluation**
   - Evaluate the model's performance using metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.
   - Calculate the RMSE for regression tasks.

8. **Insights and Recommendations**
   - Analyze the model's predictions to identify key factors influencing VM performance and resource allocation.
   - Generate actionable recommendations to optimize resource usage and improve VM performance.

9. **Visualization**
   - Create visualizations to effectively communicate the findings and recommendations.
   - Use plots such as feature importance, partial dependence plots, and performance metrics.

10. **Documentation and Reporting**
    - Document the methodology, results, and insights in a detailed report.
    - Include sections such as introduction, methodology, results, discussion, conclusion, and recommendations.

11. **Presentation Preparation**
    - Prepare a presentation summarizing the key findings and recommendations.
    - Include visual aids and clear explanations to convey the results effectively.

12. **Implementation Plan**
    - Develop an implementation plan based on the recommendations.
    - Outline steps for deploying the optimized resource allocation strategies in a real-world environment.
  
## DONE
  1. **Data Loading and Preprocessing**
   - Set up logging for tracking progress and debugging.
   - Load the dataset using the `load_data` function.
   - Handle missing values using `SimpleImputer`.
   - Normalize/standardize the data using `StandardScaler` or `PowerTransformer`.
2. **Exploratory Data Analysis (EDA)**
   - Visualize the distribution of key metrics using histograms and KDE plots.
   - Identify outliers and anomalies in the data.
   - Generate summary statistics to understand data distributions.
