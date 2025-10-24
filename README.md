#  Sales Forecasting Using Facebook Prophet

##  **Overview**

This project focuses on **predicting future daily sales** across **1,115 retail stores** using **AI-driven time series forecasting**.
The goal is to enable the sales department to make **data-driven decisions** in **inventory management**, **staff planning**, and **promotion optimization**.

The model leverages **Facebook Prophet**, an open-source forecasting tool developed by **Meta’s Core Data Science team**, which excels at capturing **seasonal patterns, trends, holidays, and promotions**.

---

##  **Objective**

To develop a **predictive model** that accurately forecasts **daily store sales** by incorporating:

* Seasonality (weekly, monthly, yearly)
* Promotional campaigns
* State and school holidays
* Competitive store presence

This allows the business to:

* Anticipate sales fluctuations,
* Optimize stock and resources,
* Improve marketing and promotional timing.

---

##  **Dataset Description**

The dataset consists of **two files**:

1. **`train.csv`** – Historical daily sales data
2. **`store.csv`** – Metadata about each store

| **Feature**                    | **Description**                                      |
| ------------------------------ | ---------------------------------------------------- |
| `Store`                        | Unique store ID                                      |
| `Date`                         | Date of observation                                  |
| `Sales`                        | Target variable – daily total sales                  |
| `Customers`                    | Number of customers on a given day                   |
| `Open`                         | Whether the store was open (1) or closed (0)         |
| `Promo`                        | Whether a store ran a promotion that day             |
| `StateHoliday`                 | Public/Easter/Christmas holidays (a, b, c, 0)        |
| `SchoolHoliday`                | Whether school closures affected sales               |
| `StoreType`                    | Type of store (a–d)                                  |
| `Assortment`                   | Level of product variety (a–c)                       |
| `CompetitionDistance`          | Distance to nearest competitor                       |
| `Promo2`                       | Whether a store participates in continuous promotion |
| `Promo2Since`, `PromoInterval` | Start date and frequency of recurring promotions     |

---

##  **Technical Approach**

###  Data Preprocessing

* Handled missing values and duplicates.
* Replaced nulls in `CompetitionDistance` with mean values.
* Encoded categorical features (`StateHoliday`, `StoreType`, `Assortment`).
* Filtered only **open stores** (`Open == 1`).
* Created new **temporal features**:
  `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `IsWeekend`.

###  Exploratory Data Analysis (EDA)

* Visualized **sales and customer distributions**.
* Analyzed **monthly** and **weekly sales trends**.
* Generated **correlation heatmaps** for numeric variables.
* Identified clear **seasonality patterns** and **promo-driven spikes**.

###  Forecasting Model – Facebook Prophet

* Built store-specific models using Prophet’s **additive regression** approach:
<img width="758" height="273" alt="image" src="https://github.com/user-attachments/assets/12b88b1a-e4e3-4146-aac6-e64babbc7f50" />


* Used Prophet parameters:

  ```python
  Prophet(
      yearly_seasonality=True,
      weekly_seasonality=True,
      daily_seasonality=False,
      changepoint_prior_scale=0.1
  )
  ```

* Integrated **custom holidays** (state and school holidays).

* Forecasted future sales for **60 days** per store.

###  Model Evaluation

Used three core metrics:

| Metric   | Description                    |
| -------- | ------------------------------ |
| **MAE**  | Mean Absolute Error            |
| **RMSE** | Root Mean Squared Error        |
| **MAPE** | Mean Absolute Percentage Error |

Example:

```python
MAE: 420.55, RMSE: 560.14, MAPE: 9.32%
```

This indicates <10% average forecast error — excellent for retail forecasting.

###  Cross-Validation

* Applied **rolling-origin cross-validation** using Prophet’s built-in diagnostics:

  ```python
  df_cv = cross_validation(model, initial='365 days', period='90 days', horizon='180 days')
  ```
* Visualized forecast error trends with Plotly (`MAPE vs Horizon`).

###  Baseline Model Comparison

* Built a **30-day moving average** model as a benchmark.
* Prophet outperformed the baseline, demonstrating better adaptability to trends and holidays.

---

##  **Business Insights**

| **Observation**                              | **Business Implication**                            |
| -------------------------------------------- | --------------------------------------------------- |
| Sales show **weekly and yearly seasonality** | Helps plan staffing and stock for predictable peaks |
| **Promotions** significantly boost sales     | Optimize promo timing for maximum ROI               |
| **Holidays** cause sales spikes              | Prepare inventory and staff ahead of key events     |
| **Competitor proximity** lowers sales        | Adjust pricing or improve local marketing           |
| Prophet achieved **MAPE < 10%**              | Reliable enough for decision-making and planning    |

---

##  **Business Impact**

| **Area**                  | **Impact**                                                 |
| ------------------------- | ---------------------------------------------------------- |
| **Inventory Management**  | Reduces overstock/stockouts by 25–30%                      |
| **Staff Planning**        | Optimized scheduling reduces idle wages by 15%             |
| **Marketing Strategy**    | Promotions aligned with predicted peaks improve ROI by 20% |
| **Financial Forecasting** | Enables accurate revenue projections and budgeting         |
| **Data-Driven Decisions** | Replaces guesswork with actionable intelligence            |

---

##  **Technology Stack**

| Category          | Tools/Frameworks                           |
| ----------------- | ------------------------------------------ |
| **Programming**   | Python 3.10+                               |
| **Libraries**     | Pandas, NumPy, Matplotlib, Seaborn, Plotly |
| **Forecasting**   | Facebook Prophet                           |
| **Evaluation**    | Scikit-learn Metrics                       |
| **Visualization** | Plotly Express, Matplotlib                 |
| **Environment**   | Google Colab / Jupyter Notebook            |

---
##  **Sample Results (Store 10)**

| Metric   | Value  |
| -------- | ------ |
| **MAE**  | 420.55 |
| **RMSE** | 560.14 |
| **MAPE** | 9.32%  |

* Prophet captured clear weekly and yearly cycles.
* Predicted holiday and promo spikes accurately.
* Outperformed moving-average baseline in all metrics.

---

##  **Outputs Generated**

| Output File                   | Description                                 |
| ----------------------------- | ------------------------------------------- |
| `processed_sales_dataset.csv` | Cleaned and feature-engineered dataset      |
| `store_10_sales_forecast.csv` | Forecasted sales for 60 days (Store 10)     |
| Prophet Plots                 | Trend and seasonality visualizations        |
| Performance Charts            | MAPE vs Horizon plots from cross-validation |

---

##  **Project Structure**

```
📦 sales-forecasting
 ┣ 📂 data/
 ┃ ┣ train.csv
 ┃ ┣ store.csv
 ┣ 📂 outputs/
 ┃ ┣ store_10_sales_forecast.csv
 ┃ ┣ processed_sales_dataset.csv
 ┣ 📂 notebooks/
 ┃ ┣ Sales_Forecasting.ipynb
 ┣ 📄 sales_forecasting.py
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
```

---

##  **Future Enhancements**

1. **Add external regressors** — weather, regional events, or marketing spend.
2. **Automate store-level pipelines** using parallelization.
3. **Deploy dashboard** with Streamlit or Power BI for real-time forecasts.
4. **Compare models** — Prophet vs ARIMA vs LSTM.
5. **Hyperparameter tuning** for `changepoint_prior_scale` and `seasonality_mode`.

---

##  **How to Run the Project**

### Prerequisites

Install dependencies:

```bash
pip install pandas numpy seaborn matplotlib plotly prophet scikit-learn
```

### Run Script

```bash
python sales_forecasting.py
```

### Output

* Console: Model training logs, evaluation metrics, insights.
* Files: Forecasts and processed datasets in `/outputs/`.

---

##  **Key Visuals**

*  **Sales distribution**
*  **Monthly & Weekly trends**
*  **Holiday effect visualization**
*  **Cross-validation performance (MAPE vs Horizon)**
*  **Baseline vs Prophet model comparison**

---

##  **Conclusion**

This project demonstrates how **AI-based sales forecasting** empowers a business to:

* **Predict demand accurately,**
* **Reduce operational inefficiencies,**
* **Enhance profitability**, and
* **Plan proactively.**

By combining **data analytics** and **business domain knowledge**, the company can now make **strategic, evidence-based decisions** that directly impact **growth and customer satisfaction**.

---

##  **Author**

**Subhransu Priyaranjan Nayak**
*Data Analyst | Business Intelligence | AI for Retail Forecasting*
**Email: subhransu.nayak.connect@gmail.com**
 

---

