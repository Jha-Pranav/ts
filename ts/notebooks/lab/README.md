
# **Baseline Models - Classical Models for Establishing a Baseline**  

## **1. Historic Average (Mean Method)**  
The **Historic Average (HA)** model predicts future values by computing the **average of all past observations**.  

### **Formula Breakdown**  
$$
\hat{y}_{t+1} = \frac{1}{t} \sum_{j=1}^{t} y_j
$$
where:  
- \( \hat{y}_{t+1} \) is the forecasted value.  
- \( y_j \) are the historical values.  
- \( t \) is the number of observations.  

### **Forecasting Process**  
1. Collect all past values up to time \( t \).  
2. Compute their arithmetic mean.  
3. Use this mean as the forecast for the next time step.  

#### **Example**  
If past values are \( [10, 12, 15, 14, 16] \):  
$$
\hat{y}_{t+1} = \frac{10 + 12 + 15 + 14 + 16}{5} = 13.4
$$

---

## **2. Naïve Model**  
The **Naïve model** assumes that the **next value will be the same as the last observed value**.  

### **Formula Breakdown**  
$$
\hat{y}_{t+1} = y_t
$$
where:  
- \( y_t \) is the last observed value.  

### **Forecasting Process**  
1. Take the last observed value \( y_t \).  
2. Use it as the forecast for the next time step.  

#### **Example**  
If \( y_t = 16 \):  
$$
\hat{y}_{t+1} = 16
$$

---

## **3. Random Walk with Drift**  
This model extends the **Naïve model** by adding a **constant drift** \( \delta \), which is the average change between observations.  

### **Formula Breakdown**  
$$
\hat{y}_{t+1} = y_t + \delta
$$
where:  
$$
\delta = \frac{1}{t-1} \sum_{j=2}^{t} (y_j - y_{j-1})
$$
- \( \delta \) represents the **average change (drift)**.  

### **Forecasting Process**  
1. Compute the average difference (drift) between consecutive values.  
2. Add this drift to the last observed value to make the prediction.  

#### **Example**  
If past values are \( [10, 12, 15, 14, 16] \):  
$$
\delta = \frac{(12-10) + (15-12) + (14-15) + (16-14)}{4} = \frac{2+3-1+2}{4} = 1.5
$$
$$
\hat{y}_{t+1} = 16 + 1.5 = 17.5
$$

---

## **4. Seasonal Naïve Model**  
Instead of using the last value, this model **uses the value from the previous season**.  

### **Formula Breakdown**  
$$
\hat{y}_{t+1} = y_{t-s}
$$
where:  
- \( s \) is the **seasonal period** (e.g., for monthly data, \( s = 12 \) for yearly seasonality).  

### **Forecasting Process**  
1. Identify the **seasonality period** \( s \).  
2. Take the value observed **one full season ago**.  
3. Use it as the forecast.  

#### **Example**  
For monthly sales data where \( s = 12 \), if January 2023 sales were **200 units**:  
$$
\hat{y}_{\text{Jan 2024}} = y_{\text{Jan 2023}} = 200
$$

---

## **5. Window Average Model**  
Instead of averaging all past values, this model **takes only the most recent \( w \) values** into account.  

### **Formula Breakdown**  
$$
\hat{y}_{t+1} = \frac{1}{w} \sum_{j=t-w+1}^{t} y_j
$$
where:  
- \( w \) is the **window size** (number of recent observations to consider).  

### **Forecasting Process**  
1. Choose a window size \( w \).  
2. Take the last \( w \) values and compute their mean.  
3. Use this mean as the forecast.  

#### **Example**  
If the last **3** observations are \( [14, 16, 18] \) and \( w = 3 \):  
$$
\hat{y}_{t+1} = \frac{14 + 16 + 18}{3} = 16
$$

---

## **6. Seasonal Window Average Model**  
This model improves **Seasonal Naïve** by averaging multiple past seasonal values instead of using just one.  

### **Formula Breakdown**  
$$
\hat{y}_{t+1} = \frac{1}{N} \sum_{k=1}^{N} y_{t - k \cdot s}
$$
where:  
- \( N \) is the number of past seasons to consider.  
- \( s \) is the **seasonality period**.  

### **Forecasting Process**  
1. Choose the number of past seasons \( N \).  
2. Take the values from **previous seasons**.  
3. Compute their average.  
4. Use this average as the forecast.  

#### **Example**  
For quarterly sales data (\( s = 4 \)) with the last **2** seasonal values:  
If past Q1 sales were **[200, 220]**:  
$$
\hat{y}_{Q1, t+1} = \frac{200 + 220}{2} = 210
$$

---

## **Comparison of Baseline Models**

| Model                      | Captures Trends? | Captures Seasonality? | Complexity |
|----------------------------|-----------------|-----------------------|------------|
| **Historic Average**        | ❌ No           | ❌ No                  | ⭐ Simple  |
| **Naïve**                  | ❌ No           | ❌ No                  | ⭐ Simple  |
| **Random Walk with Drift**  | ✅ Yes          | ❌ No                  | ⭐⭐ Medium |
| **Seasonal Naïve**          | ❌ No           | ✅ Yes                 | ⭐ Simple  |
| **Window Average**          | ✅ Partial      | ❌ No                  | ⭐⭐ Medium |
| **Seasonal Window Average** | ✅ Partial      | ✅ Yes                 | ⭐⭐ Medium |

---
---

# **Exponential Smoothing Models with Examples**  

Exponential Smoothing methods predict future values by giving **more weight to recent observations** while exponentially decreasing the influence of older values.  

## **1. Simple Exponential Smoothing (SES)**  
The **SES model** is best for **data without trend or seasonality**. It smooths past values using an **exponentially weighted average**.  

### **Formula**  
$$
\hat{y}_{t+1} = \alpha y_t + (1 - \alpha) \hat{y}_t
$$  
where:  
- \( \alpha \) is the **smoothing parameter** (\( 0 \leq \alpha \leq 1 \)).  
- \( \hat{y}_t \) is the previous forecast.  
- \( y_t \) is the actual observed value.  

### **Example Calculation**  
Consider a demand forecast for **monthly sales**:  
- Given: \( y_t = 100 \) (current observation), \( \hat{y}_t = 90 \) (previous forecast), \( \alpha = 0.2 \).  
- Using SES formula:  
  $$
  \hat{y}_{t+1} = (0.2 \times 100) + (0.8 \times 90) = 92
  $$  
Thus, the forecast for next month is **92** units.  

---

## **2. Simple Exponential Smoothing (Optimized)**  
This method is the **same as SES**, but the smoothing parameter \( \alpha \) is **automatically optimized**.  

### **Example**  
Instead of manually choosing \( \alpha = 0.2 \), an optimization algorithm finds the best value (e.g., \( \alpha = 0.35 \)) to minimize forecasting errors.  

---

## **3. Seasonal Exponential Smoothing**  
**Handles seasonality** by applying different smoothing for each season.  

### **Formula**  
For **multiplicative seasonality**:  
$$
\hat{y}_{t+1} = (S_{t-s+1}) \times \left( \alpha y_t + (1 - \alpha) \hat{y}_t \right)
$$  
where:  
- \( S_t \) is the **seasonal factor**.  
- \( s \) is the **season length**.  

### **Example Calculation**  
A retailer records **quarterly sales** with seasonality:  
- **Q1 (last year)**: **500 units**  
- Seasonal factor \( S_{t-s} = 1.1 \)  
- \( \alpha = 0.3 \), \( y_t = 550 \), \( \hat{y}_t = 520 \).  

Using the formula:  
$$
\hat{y}_{t+1} = (1.1) \times [(0.3 \times 550) + (0.7 \times 520)]
$$  
$$
\hat{y}_{t+1} = (1.1) \times [165 + 364] = 580.9
$$  
So the forecast for **next quarter** is **581 units**.  

---

## **4. Seasonal Exponential Smoothing (Optimized)**  
Same as above, but the seasonal parameters (\( \alpha \), \( S_t \)) are **automatically optimized** to minimize error.  

---

## **5. Holt’s Linear Trend Model**  
Extends SES to handle **data with trend**.  

### **Formula**  
Level equation:  
$$
l_t = \alpha y_t + (1 - \alpha)(l_{t-1} + b_{t-1})
$$  
Trend equation:  
$$
b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1}
$$  
Forecast equation:  
$$
\hat{y}_{t+h} = l_t + h b_t
$$  
where:  
- \( l_t \) is the **level** (smoothed value).  
- \( b_t \) is the **trend** (slope).  

### **Example Calculation**  
A company records **annual revenue** increasing by **$500K per year**:  
- Last revenue: **$2M**  
- Previous trend: **$400K per year**  
- \( \alpha = 0.4 \), \( \beta = 0.3 \).  

Compute level and trend:  
$$
l_t = (0.4 \times 2.5M) + (0.6 \times (2M + 0.4M)) = 2.24M
$$  
$$
b_t = (0.3 \times (2.24M - 2M)) + (0.7 \times 0.4M) = 0.376M
$$  
Forecast for **3 years ahead**:  
$$
\hat{y}_{t+3} = 2.24M + (3 \times 0.376M) = 3.368M
$$  
Thus, the projected revenue for **3 years ahead** is **$3.37M**.  

---

## **6. Holt-Winters Method (Triple Exponential Smoothing)**  
Handles both **trend and seasonality**.  

### **Formula**  
Level:  
$$
l_t = \alpha \frac{y_t}{S_{t-s}} + (1 - \alpha)(l_{t-1} + b_{t-1})
$$  
Trend:  
$$
b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1}
$$  
Seasonality:  
$$
S_t = \gamma \frac{y_t}{l_t} + (1 - \gamma) S_{t-s}
$$  
Forecast:  
$$
\hat{y}_{t+h} = (l_t + h b_t) S_{t+h-s}
$$  

### **Example Calculation**  
Sales data shows a **monthly trend** of **$2K per month** and a **seasonality of 1.2 in December**:  
- \( l_t = 50K \), \( b_t = 2K \), \( S_{t-s} = 1.2 \).  
- Forecast **6 months ahead**:  
$$
\hat{y}_{t+6} = (50K + (6 \times 2K)) \times 1.2 = 67.2K
$$  
Thus, **projected sales for December** is **$67.2K**.  

---

## **7. AutoETS (Automatic Exponential Smoothing - ETS Model)**  
**Automatically selects the best ETS (Error, Trend, Seasonality) model** based on data.  

### **Example Usage**  
A **demand forecasting system** can use AutoETS to:  
1. Identify if **trend** exists.  
2. Detect if **seasonality** is present.  
3. Choose between **additive/multiplicative models**.  

No manual selection is needed—AutoETS **chooses the best** smoothing parameters.  

---

## **8. AutoCES (Automatic Complex Exponential Smoothing)**  
**Handles nonlinear trends and multiple seasonality**.  

### **Example Use Case**  
A ride-hailing app wants to forecast demand based on:  
- **Daily trends (rush hour peaks).**  
- **Weekly seasonality (weekend surge).**  
- **Yearly holiday patterns.**  

AutoCES can automatically model all these factors without manual intervention.  

---

## **Comparison of Exponential Smoothing Models**

| Model                          | Handles Trend? | Handles Seasonality? | Optimized? | Best For |
|--------------------------------|---------------|----------------------|------------|----------|
| **Simple Exponential Smoothing (SES)** | ❌ No | ❌ No | ❌ No | Short-term forecasting |
| **SES Optimized**              | ❌ No | ❌ No | ✅ Yes | Short-term forecasting |
| **Seasonal Exponential Smoothing** | ❌ No | ✅ Yes | ❌ No | Seasonal data |
| **Seasonal SES Optimized**      | ❌ No | ✅ Yes | ✅ Yes | Seasonal data |
| **Holt’s Linear Model**         | ✅ Yes | ❌ No | ❌ No | Trend-based series |
| **Holt-Winters**               | ✅ Yes | ✅ Yes | ❌ No | Seasonal & trend-based data |
| **AutoETS**                     | ✅ Yes | ✅ Yes | ✅ Yes | Automated forecasting |
| **AutoCES**                     | ✅ Yes | ✅ Yes | ✅ Yes | Complex seasonal patterns |

---
---

# **ARIMA Family Models**  

The **ARIMA (AutoRegressive Integrated Moving Average) family** of models is widely used for time series forecasting, particularly when the data has trends, seasonality, and autocorrelation.  

## **1. AutoARIMA (Automatic ARIMA)**  
AutoARIMA **automatically finds the best ARIMA model** by optimizing its parameters:  
- **AR (AutoRegressive) component**: Captures past values' influence.  
- **I (Integrated) component**: Handles trends by differencing.  
- **MA (Moving Average) component**: Captures past forecast errors.  

### **Formula**  
A general ARIMA(\(p,d,q\)) model is expressed as:  
$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t
$$  
where:  
- \( p \) = Number of **autoregressive** (AR) terms.  
- \( d \) = Number of **differences** applied to remove trend.  
- \( q \) = Number of **moving average** (MA) terms.  
- \( c \) = Constant.  
- \( \phi_i \) = AR coefficients.  
- \( \theta_j \) = MA coefficients.  
- \( \epsilon_t \) = White noise error term.  

### **Example Calculation**  
A company tracks **monthly product demand**:  
- **Observed values**: \( y_t = [120, 130, 125, 140, 150] \).  
- **AutoARIMA detects**: \( p=1, d=1, q=1 \) as best fit → ARIMA(1,1,1).  

The AutoARIMA **automatically selects**:  
$$
y_t = 0.5 y_{t-1} + 0.3 \epsilon_{t-1} + \epsilon_t
$$  

Using previous demand values:  
- **\( y_{t-1} = 140 \)**  
- **\( \epsilon_{t-1} = -5 \)** (error from previous forecast)  

Forecasting **next month**:  
$$
y_{t+1} = (0.5 \times 140) + (0.3 \times -5) = 67.5
$$  
Thus, the **forecasted demand for next month** is **~135 units**.  

---

## **2. AutoRegressive (AR) Model**  
The **AutoRegressive (AR) model** predicts future values based on a **linear combination of past values**.  

### **Formula**  
An **AR(p)** model is written as:  
$$
y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \epsilon_t
$$  
where:  
- \( p \) = Number of past observations considered.  
- \( \phi_i \) = Coefficients that determine how past values influence the forecast.  
- \( c \) = Constant.  
- \( \epsilon_t \) = White noise error term.  

### **Example Calculation**  
A stock market analyst uses an **AR(2) model** to forecast stock prices:  
- Past values: **\( y_{t-1} = 110 \), \( y_{t-2} = 105 \)**.  
- Given coefficients: \( \phi_1 = 0.7 \), \( \phi_2 = 0.2 \).  

Using the AR formula:  
$$
y_{t+1} = (0.7 \times 110) + (0.2 \times 105) = 86.9
$$  
Thus, the **forecasted stock price** for next time step is **~112.5**.  

---

## **Comparison of ARIMA Family Models**  

| Model        | Handles Trend? | Handles Seasonality? | Optimized? | Best For |
|-------------|---------------|----------------------|------------|----------|
| **AutoARIMA**   | ✅ Yes | ✅ Yes | ✅ Yes | General time series forecasting |
| **AutoRegressive (AR)** | ✅ Yes | ❌ No | ❌ No | Forecasting when past values influence future |

---


---

# **Theta Family Models**  

The **Theta model** is a simple yet powerful time series forecasting method that **decomposes the original series into multiple Theta lines**, applies different smoothing techniques, and combines the forecasts. It is particularly effective for **short-term forecasting** and **trend handling**.

---

## **1. Theta Model**  
The **basic Theta model** works by decomposing a time series into multiple versions, called **Theta lines**, and combining their forecasts.  

### **Formula**  
The original time series \( y_t \) is decomposed into **multiple Theta lines**:  
$$
y_t^{(\theta)} = \theta y_t + (1 - \theta) l_t
$$  
where:  
- \( \theta \) is the **Theta coefficient** (controls smoothing).  
- \( l_t \) is the **local trend component**, usually computed as a **linear regression slope**.  

### **Forecasting Equation**  
The final forecast is obtained by averaging different Theta lines:  
$$
\hat{y}_{t+h} = \frac{1}{m} \sum_{i=1}^{m} \hat{y}_{t+h}^{(\theta_i)}
$$  
where:  
- \( m \) = Number of Theta lines used.  

### **Example Calculation**  
A company tracks **monthly website visits**:  
- **Observed data**: \( y_t = [500, 520, 540, 560, 580] \).  
- **Theta values used**: \( \theta_1 = 0, \theta_2 = 2 \).  

For each Theta line:  
1. **Theta(0) line** (trend-only): Linear regression estimates a **slope of 20**.  
2. **Theta(2) line** (twice the trend): **Doubles the slope to 40**.  

Final forecast for **next month**:  
$$
\hat{y}_{t+1} = \frac{1}{2} [(580 + 20) + (580 + 40)] = 610
$$  
Thus, the **forecasted website visits next month** is **610 visits**.  

---

## **2. Optimized Theta**  
This is an **improved Theta model** where the system **automatically optimizes the Theta parameters** to minimize forecasting error.  

### **Example Use Case**  
Instead of manually choosing \( \theta = [0, 2] \), the algorithm selects the best Theta values, e.g., **\( \theta = [0.3, 1.8] \)**, for more accurate predictions.  

---

## **3. Dynamic Theta**  
A **variant of the Theta model** where Theta parameters are **dynamically adjusted over time** instead of being fixed.  

### **Example Use Case**  
A **stock price forecasting** system uses **Dynamic Theta** to adjust its trend estimation based on the latest market fluctuations.  

---

## **4. Dynamic Optimized Theta**  
This is a **combination of Dynamic Theta and Optimized Theta**, where:  
- **Theta values dynamically adjust** as new data comes in.  
- **An optimization algorithm selects the best Theta coefficients**.  

### **Example Use Case**  
A **weather prediction system** uses **Dynamic Optimized Theta** to improve accuracy by continuously adjusting the smoothing parameters as new temperature data arrives.  

---

## **5. AutoTheta**  
An **automated version of the Theta model** that selects:  
- **Best Theta values** automatically.  
- **Optimal number of Theta lines**.  
- **Best combination of short-term and long-term trends**.  

### **Example Use Case**  
A **sales forecasting tool** for an e-commerce business uses **AutoTheta** to automatically determine the best trend and seasonality components for product demand forecasting.  

---

## **Comparison of Theta Family Models**  

| Model                      | Handles Trend? | Handles Seasonality? | Optimized? | Dynamic? | Best For |
|----------------------------|---------------|----------------------|------------|----------|----------|
| **Theta**                  | ✅ Yes | ❌ No | ❌ No | ❌ No | Basic trend-based forecasting |
| **Optimized Theta**        | ✅ Yes | ❌ No | ✅ Yes | ❌ No | More accurate trend-based forecasts |
| **Dynamic Theta**          | ✅ Yes | ❌ No | ❌ No | ✅ Yes | Forecasting with real-time trend updates |
| **Dynamic Optimized Theta**| ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | Real-time adaptive forecasting |
| **AutoTheta**              | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | Fully automated trend and seasonality forecasting |

---

---

# **Multiple Seasonality Models with Examples**  

Time series data often has **multiple seasonal cycles**, such as **hourly, daily, and yearly trends** in electricity consumption or website traffic. These models **handle complex seasonal patterns** better than traditional methods like ARIMA.

---

## **1. MSTL (Multiple Seasonal-Trend Decomposition using LOESS)**  
MSTL is a **decomposition-based model** that breaks a time series into:  
- **Multiple seasonal components** (e.g., weekly, monthly, yearly).  
- **Trend component**.  
- **Residual (random) component**.  

### **Formula**  
Given a time series \( y_t \), MSTL decomposes it as:  
$$
y_t = \sum_{i=1}^{m} S_{i,t} + T_t + R_t
$$  
where:  
- \( S_{i,t} \) = Seasonal components for different cycles (e.g., **daily, weekly, yearly**).  
- \( T_t \) = Trend component (smoothed using **LOESS regression**).  
- \( R_t \) = Residual (random noise).  

### **Example Calculation**  
A **daily sales dataset** has:  
- **Weekly seasonality** (\( S_1 \)).  
- **Monthly seasonality** (\( S_2 \)).  

MSTL extracts:  
- \( S_1 \) = **Repeating weekly sales pattern**.  
- \( S_2 \) = **Monthly holiday demand spikes**.  
- \( T_t \) = **Overall increasing trend** in sales.  

MSTL then **reconstructs** the forecast using these components.  

---

## **2. AutoMFLES (Automatic Multiple Frequency Least Squares Estimation)**  
AutoMFLES automatically detects **multiple seasonalities** and applies **Least Squares Estimation (LSE)** to model them.  

### **Formula**  
AutoMFLES fits a regression model with multiple seasonal dummies:  
$$
y_t = \beta_0 + \sum_{i=1}^{m} \beta_i S_{i,t} + \epsilon_t
$$  
where:  
- \( \beta_i \) = Estimated seasonal coefficients.  
- \( S_{i,t} \) = Seasonal dummy variables for different cycles.  
- \( \epsilon_t \) = Error term.  

### **Example Calculation**  
For **hourly temperature data**, AutoMFLES detects:  
- **Daily temperature cycle** (cooler at night, warmer in the afternoon).  
- **Yearly seasonality** (summer vs. winter).  

It fits a regression model to **capture both seasonal effects** and **forecast future temperatures accurately**.  

---

## **3. AutoTBATS**  
TBATS is a powerful model for handling **multiple seasonalities, trend, and Box-Cox transformations**.  

### **TBATS Components**  
TBATS is an acronym:  
- **T**: Trigonometric seasonality handling.  
- **B**: Box-Cox transformation (to stabilize variance).  
- **A**: ARMA errors (to model short-term correlations).  
- **T**: Trend handling.  
- **S**: Seasonal decomposition (using Fourier series).  

### **Formula**  
TBATS models seasonalities using Fourier terms:  
$$
y_t = l_t + \phi b_{t-1} + \sum_{j=1}^{m} \sum_{k=1}^{K_j} s_{j,k,t} + \epsilon_t
$$  
where:  
- \( l_t \) = Local level (trend).  
- \( b_t \) = Slope (drift).  
- \( \phi \) = Damping parameter.  
- \( s_{j,k,t} \) = Seasonal components modeled using Fourier terms.  

### **Example Calculation**  
For **monthly airline passenger data**, TBATS detects:  
- **Annual seasonality** (more travel in summer, holidays).  
- **Biannual fluctuations** (peak in June, smaller peak in December).  
- **Trend** (steady increase over years).  

TBATS combines all components to **forecast future passenger numbers**.  

---

## **Comparison of Multiple Seasonality Models**  

| Model      | Handles Multiple Seasonality? | Handles Trend? | Uses Decomposition? | Best For |
|-----------|------------------------------|---------------|---------------------|----------|
| **MSTL**      | ✅ Yes | ✅ Yes | ✅ Yes | Complex seasonal patterns with trend |
| **AutoMFLES** | ✅ Yes | ❌ No  | ❌ No  | Multiple seasonalities in regression models |
| **AutoTBATS** | ✅ Yes | ✅ Yes | ❌ No  | Seasonal data with trend & irregular cycles |

---

---

# **GARCH and ARCH Models with Examples**  

ARCH (Autoregressive Conditional Heteroskedasticity) and GARCH (Generalized ARCH) models are used for **modeling time series volatility**, such as **financial returns, stock market data, and economic indicators**.  

These models capture how **variance (volatility) changes over time**, making them useful for risk analysis and forecasting market fluctuations.

---

## **1. ARCH (Autoregressive Conditional Heteroskedasticity) Model**  
The **ARCH model** captures **time-varying volatility** by modeling variance as a function of past squared residuals.  

### **Formula**  
An **ARCH(q)** model expresses conditional variance as:  
$$
\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2
$$  
where:  
- \( \sigma_t^2 \) = Conditional variance at time \( t \).  
- \( \alpha_0 \) = Constant (must be **positive**).  
- \( \alpha_i \) = Coefficients that determine how past squared errors \( \epsilon_{t-i}^2 \) impact volatility.  
- \( q \) = Number of past periods considered.  
- \( \epsilon_t \) = White noise error term.  

### **Example Calculation**  
A financial analyst is forecasting the **volatility of daily stock returns**:  
- **Past squared errors**: \( \epsilon_{t-1}^2 = 0.02 \), \( \epsilon_{t-2}^2 = 0.01 \).  
- **ARCH(2) model**:  
  - \( \alpha_0 = 0.01 \), \( \alpha_1 = 0.4 \), \( \alpha_2 = 0.3 \).  

Forecasting **next day's volatility**:  
$$
\sigma_t^2 = 0.01 + (0.4 \times 0.02) + (0.3 \times 0.01)
$$  
$$
\sigma_t^2 = 0.01 + 0.008 + 0.003 = 0.021
$$  
Thus, the **predicted variance for the next period** is **0.021**.

---

## **2. GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model**  
The **GARCH model** extends ARCH by incorporating **past volatility** into the model, making it more stable and realistic for financial data.  

### **Formula**  
A **GARCH(p, q)** model expresses variance as:  
$$
\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2
$$  
where:  
- \( \sigma_t^2 \) = Conditional variance at time \( t \).  
- \( \alpha_0 \) = Constant.  
- \( \alpha_i \) = Coefficients for past squared errors (like ARCH).  
- \( \beta_j \) = Coefficients for past variances.  
- \( p \) = Number of past variances considered.  
- \( q \) = Number of past squared errors considered.  

### **Example Calculation**  
A hedge fund is modeling **daily exchange rate volatility** using **GARCH(1,1)**:  
- **Past variance**: \( \sigma_{t-1}^2 = 0.015 \).  
- **Past squared error**: \( \epsilon_{t-1}^2 = 0.02 \).  
- **Model parameters**:  
  - \( \alpha_0 = 0.005 \), \( \alpha_1 = 0.4 \), \( \beta_1 = 0.5 \).  

Forecasting **next day's volatility**:  
$$
\sigma_t^2 = 0.005 + (0.4 \times 0.02) + (0.5 \times 0.015)
$$  
$$
\sigma_t^2 = 0.005 + 0.008 + 0.0075 = 0.0205
$$  
Thus, the **predicted variance for the next period** is **0.0205**.

---

## **Comparison of ARCH and GARCH Models**  

| Model    | Captures Past Errors? | Captures Past Variance? | Best For |
|----------|----------------------|-------------------------|----------|
| **ARCH**  | ✅ Yes | ❌ No | Short-term volatility forecasting |
| **GARCH** | ✅ Yes | ✅ Yes | Long-term, stable volatility modeling |

---

## **When to Use ARCH vs. GARCH?**  
- **Use ARCH** when **volatility clusters** appear, but only **recent shocks matter**.  
- **Use GARCH** when **both past shocks and past volatility influence future volatility**, making it **more robust for long-term forecasting**.  

---

---

# **Sparse or Intermittent Demand Forecasting Models**  

Intermittent demand occurs when **demand is irregular**, with many periods of **zero demand**, making traditional forecasting methods (like ARIMA or Exponential Smoothing) **ineffective**. These specialized models handle **sporadic demand patterns**, commonly found in **inventory management, spare parts forecasting, and retail demand prediction**.

---

## **1. ADIDA (Aggregate-Disaggregate Intermittent Demand Approach)**  
ADIDA transforms **intermittent demand** into a **continuous time series** by aggregating data into larger time buckets.  

### **Forecasting Approach**  
1. **Aggregate demand** over fixed intervals (e.g., weekly instead of daily).  
2. Apply **traditional forecasting methods** (like Exponential Smoothing or ARIMA) on the aggregated series.  
3. **Disaggregate** the forecast back to the original time scale.  

### **Example Calculation**  
- **Daily demand data**: [0, 0, 10, 0, 0, 20, 0, 0, 30].  
- **Aggregated into weekly buckets**: [10, 20, 30].  
- **Forecasting weekly demand**: Predict **next week’s demand as 25**.  
- **Disaggregate** back to daily: Assign demand **proportionally or evenly over the week**.  

This method smooths out **zero values** and makes forecasting easier.  

---

## **2. Croston’s Method (Classic)**  
Croston’s method **separately models demand size and demand intervals** using Exponential Smoothing.  

### **Formula**  
Croston’s forecast is given by:  
$$
\hat{y}_{t+1} = \frac{\hat{z}_t}{\hat{p}_t}
$$  
where:  
- \( \hat{z}_t \) = **Smoothed demand size** using Exponential Smoothing:  
  $$
  \hat{z}_t = \alpha z_t + (1 - \alpha) \hat{z}_{t-1}
  $$  
- \( \hat{p}_t \) = **Smoothed inter-arrival time (interval between non-zero demands)**:  
  $$
  \hat{p}_t = \beta p_t + (1 - \beta) \hat{p}_{t-1}
  $$  
- \( \alpha, \beta \) = Smoothing parameters.  

### **Example Calculation**  
A warehouse tracks **monthly spare part sales**:  
- Demand occurs every **3 months** with sizes: **[0, 0, 5, 0, 0, 10, 0, 0, 15]**.  
- **Average demand size** = **10**.  
- **Average interval** = **3 months**.  

Forecast:  
$$
\hat{y}_{t+1} = \frac{10}{3} = 3.33
$$  
Thus, the **expected demand per month is 3.33 units**.  

---

## **3. Croston Optimized**  
An improved version of Croston’s method where **smoothing parameters (\(\alpha, \beta\)) are optimized** using error minimization.  

### **Example Use Case**  
- Instead of **fixed smoothing factors** like 0.1 or 0.2, an **optimization algorithm** (e.g., grid search) selects the best **\(\alpha\) and \(\beta\)** to minimize forecasting error.  

---

## **4. Croston SBA (Syntetos-Boylan Approximation)**  
A bias-corrected version of Croston’s method, adjusting for **overestimation bias**.  

### **Formula**  
$$
\hat{y}_{t+1} = \frac{\hat{z}_t}{\hat{p}_t} \times (1 - \frac{\beta}{2})
$$  
where \( \beta \) is the smoothing factor for inter-arrival time.  

### **Example Calculation**  
Using the **same data as Croston’s method** with \( \beta = 0.2 \):  
$$
\hat{y}_{t+1} = \frac{10}{3} \times (1 - 0.1) = 3
$$  
Thus, **SBA gives a slightly lower forecast than Croston Classic** to correct for bias.  

---

## **5. IMAPA (Intermittent Multiple Aggregation Prediction Algorithm)**  
IMAPA **combines multiple ADIDA forecasts** by aggregating data at different levels and averaging the predictions.  

### **Forecasting Approach**  
1. Aggregate data at **multiple levels** (e.g., **weekly, bi-weekly, and monthly**).  
2. Apply **Croston’s or other forecasting models** at each level.  
3. Combine forecasts using a weighted average.  

### **Example Use Case**  
A **medical supply company** tracks demand for **rare medications**:  
- **Weekly demand**: Croston predicts **3.2 units per week**.  
- **Bi-weekly demand**: Croston predicts **6.5 units per two weeks**.  
- **Monthly demand**: Croston predicts **12 units per month**.  

Final **IMAPA forecast**:  
$$
\hat{y}_{t+1} = \frac{3.2 + 6.5/2 + 12/4}{3} = 3.1
$$  
IMAPA provides a **more stable** prediction by combining multiple aggregation levels.  

---

## **6. TSB (Teunter-Syntetos-Babai) Model**  
TSB is an alternative **exponential smoothing model** for intermittent demand that **updates both demand probability and demand size separately**.  

### **Formula**  
1. **Probability of demand occurrence**:  
   $$
   \hat{\pi}_t = \gamma I_t + (1 - \gamma) \hat{\pi}_{t-1}
   $$  
   where \( I_t = 1 \) if there’s demand, otherwise \( 0 \).  

2. **Demand size forecast**:  
   $$
   \hat{z}_t = \alpha y_t + (1 - \alpha) \hat{z}_{t-1}
   $$  

3. **Final Forecast**:  
   $$
   \hat{y}_{t+1} = \hat{\pi}_t \times \hat{z}_t
   $$  

### **Example Calculation**  
- **Observed demand pattern**: [0, 0, 5, 0, 0, 10, 0, 0, 15].  
- **Initial probability of demand (\(\pi_t\))** = 0.33.  
- **Smoothed demand size (\(\hat{z}_t\))** = 10.  

Final forecast:  
$$
\hat{y}_{t+1} = 0.33 \times 10 = 3.3
$$  
TSB **better captures demand probability** than Croston’s method.  

---

## **Comparison of Sparse Demand Models**  

| Model          | Handles Demand Size? | Handles Demand Interval? | Bias Correction? | Best For |
|---------------|----------------------|--------------------------|------------------|----------|
| **ADIDA**     | ✅ Yes | ✅ Yes | ❌ No | Aggregating intermittent demand |
| **Croston**   | ✅ Yes | ✅ Yes | ❌ No | Basic intermittent forecasting |
| **Croston Optimized** | ✅ Yes | ✅ Yes | ❌ No | Same as Croston but optimized |
| **Croston SBA** | ✅ Yes | ✅ Yes | ✅ Yes | Bias-corrected Croston |
| **IMAPA**     | ✅ Yes | ✅ Yes | ✅ Yes | Multi-level aggregation |
| **TSB**       | ✅ Yes | ✅ Yes | ✅ Yes | Best for low-frequency demand |

---

## **When to Use Each Model?**  
- **Use ADIDA** if demand is extremely sporadic and needs aggregation.  
- **Use Croston** for simple intermittent demand forecasting.  
- **Use Croston SBA** if Croston’s method overestimates demand.  
- **Use IMAPA** for more stable, multi-scale forecasts.  
- **Use TSB** if the probability of demand occurrence **varies significantly over time**.  

---
---

# **Time Series Forecasting Models & Their Best Use Cases**  

| **Model**  | **Best for TS Characteristics** | **Feature Engineering** | **Real-World Use Case** |
|------------|--------------------------------|-------------------------|--------------------------|
| **Baseline Models** |
| **HistoricAverage** | Stable TS with no trend or seasonality | None | Estimating daily website visits |
| **Naive** | Strong trend, no seasonality | None | Stock price forecasting |
| **RandomWalkWithDrift** | Linear trend, no seasonality | Detrending | Predicting long-term inflation |
| **SeasonalNaive** | Strong seasonality | Seasonal decomposition | Monthly energy consumption |
| **WindowAverage** | Short-term forecasting, noisy TS | Moving average smoothing | Predicting sales in a supermarket chain |
| **SeasonalWindowAverage** | Seasonal patterns with short-term variations | Lag-based features | Forecasting ice cream sales based on weather |
| **Exponential Smoothing Models** |
| **SimpleExponentialSmoothing** | No trend or seasonality, short-term | Smoothing, differencing | Sales forecasting for a small store |
| **SimpleExponentialSmoothingOptimized** | Noisy data with weak patterns | Hyperparameter tuning | Demand forecasting for perishable goods |
| **SeasonalExponentialSmoothing** | Strong seasonality | Seasonal decomposition | Holiday demand prediction in e-commerce |
| **SeasonalExponentialSmoothingOptimized** | Seasonal fluctuations | Fourier transforms | Retail sales forecasting |
| **Holt** | Linear trend | Trend decomposition | Forecasting company revenue growth |
| **HoltWinters** | Trend + Seasonality | Lagged features, seasonality encoding | Airline passenger prediction |
| **AutoETS** | Complex trends & seasonality | Automated hyperparameter tuning | Monthly hotel bookings |
| **AutoCES** | Data with non-linear trends | Feature interactions | Cryptocurrency price prediction |
| **ARIMA Family** |
| **AutoARIMA** | Stationary data, trend removal required | Differencing, stationarity tests | GDP growth forecasting |
| **AutoRegressive (AR)** | Strong autocorrelation | Lag selection | Forecasting electricity demand |
| **Theta Family** |
| **Theta** | TS with changing trends | Trend decomposition | Predicting internet traffic for a news website |
| **OptimizedTheta** | Unstable trends | Automated parameter selection | Demand forecasting for fast-moving consumer goods |
| **DynamicTheta** | TS with external influences | Exogenous variable incorporation | Supply chain forecasting under dynamic demand |
| **DynamicOptimizedTheta** | Multiple trend regimes | Bayesian optimization | Traffic congestion prediction |
| **AutoTheta** | Long-term forecasting with trend shifts | Feature scaling | Global temperature forecasting |
| **Multiple Seasonalities** |
| **MSTL (Multiple Seasonal-Trend Decomposition)** | TS with multiple seasonal patterns | Decomposition, Fourier transforms | Predicting retail sales with holiday effects |
| **AutoMFLES (Automatic Multi-Frequency Level Estimation Smoothing)** | High-frequency seasonal fluctuations | Wavelet transformations | Power grid load forecasting |
| **AutoTBATS** | Complex seasonality, multiple cycles | Fourier transforms, Lag-based features | Weather prediction with daily and yearly cycles |
| **GARCH & ARCH Models** |
| **ARCH** | High-volatility time series | Volatility clustering analysis | Risk modeling in financial markets |
| **GARCH** | Time-dependent volatility | Feature scaling | Forecasting stock market fluctuations |
| **Sparse/Intermittent Demand Models** |
| **ADIDA** | Sparse demand data | Data aggregation | Forecasting spare parts demand |
| **CrostonClassic** | Irregular demand intervals | Exponential smoothing | Demand prediction for slow-moving inventory |
| **CrostonOptimized** | Intermittent demand with varying trends | Hyperparameter tuning | Predicting hospital supply usage |
| **CrostonSBA** | Bias correction in intermittent demand | Bias reduction | Military equipment forecasting |
| **IMAPA** | Multi-scale intermittent demand | Multi-resolution decomposition | Predicting seasonal clothing sales |
| **TSB** | Demand probability forecasting | Probability-based feature extraction | Predicting emergency room visits |

---

## **Key Insights from the Table**  
1. **Trend & Seasonality**  
   - **Use Exponential Smoothing (Holt, Holt-Winters) or ARIMA** for TS with **trend or seasonality**.  
   - **Use AutoETS or AutoTBATS** for **complex seasonal variations**.  

2. **Volatility & Risk Modeling**  
   - **Use ARCH/GARCH** when **variance changes over time** (e.g., stock markets, economic forecasting).  

3. **Sparse Demand Forecasting**  
   - **Use Croston’s Methods (Classic, SBA, IMAPA, TSB) or ADIDA** for forecasting **low-frequency sales data**.  

4. **Multiple Seasonality Handling**  
   - **MSTL, AutoTBATS, AutoMFLES** are best for **data with overlapping seasonal patterns** (e.g., power grid, weather, retail).  

5. **Short vs. Long-Term Forecasting**  
   - **AutoTheta, Holt-Winters, ARIMA** are better for **long-term** predictions.  
   - **WindowAverage, Naïve, TSB** are better for **short-term** predictions.  

---
