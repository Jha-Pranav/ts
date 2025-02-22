# Baseline Models - Classical models for establishing baseline.

## **1. Historic Average (Mean Method)**  
The **Historic Average (HA)** model predicts future values by computing the **average of all past observations**.

### **Formula Breakdown**  
\[
\hat{y}_{t+1} = \frac{1}{t} \sum_{j=1}^{t} y_j
\]
where:
- \( \hat{y}_{t+1} \) is the forecasted value.
- \( y_j \) are the historical values.
- \( t \) is the number of observations.
- \( \frac{1}{t} \sum_{j=1}^{t} y_j \) computes the **mean** of past values.

### **Forecasting Process**  
1. Collect all past values up to time \( t \).
2. Compute their arithmetic mean.
3. Use this mean as the forecast for the next time step.

#### **Example**
If past values are \( [10, 12, 15, 14, 16] \),  
\[
\hat{y}_{t+1} = \frac{10 + 12 + 15 + 14 + 16}{5} = 13.4
\]

---

## **2. Naïve Model**  
The **Naïve model** assumes that the **next value will be the same as the last observed value**.

### **Formula Breakdown**  
\[
\hat{y}_{t+1} = y_t
\]
where:
- \( y_t \) is the last observed value.
- The model simply **copies** the last known value.

### **Forecasting Process**  
1. Take the last observed value \( y_t \).
2. Use it as the forecast for the next time step.

#### **Example**
If \( y_t = 16 \),  
\[
\hat{y}_{t+1} = 16
\]

---

## **3. Random Walk with Drift**  
This model extends the **Naïve model** by adding a **constant drift** \( \delta \), which is the average change between observations.

### **Formula Breakdown**  
\[
\hat{y}_{t+1} = y_t + \delta
\]
where:
- \( \delta = \frac{1}{t-1} \sum_{j=2}^{t} (y_j - y_{j-1}) \) is the average change (drift).
- The model assumes a gradual **increasing or decreasing trend**.

### **Forecasting Process**  
1. Compute the average difference (drift) between consecutive values.
2. Add this drift to the last observed value to make the prediction.

#### **Example**
If past values are \( [10, 12, 15, 14, 16] \),  
\[
\delta = \frac{(12-10) + (15-12) + (14-15) + (16-14)}{4} = \frac{2+3-1+2}{4} = 1.5
\]
\[
\hat{y}_{t+1} = 16 + 1.5 = 17.5
\]

---

## **4. Seasonal Naïve Model**  
Instead of using the last value, this model **uses the value from the previous season**.

### **Formula Breakdown**  
\[
\hat{y}_{t+1} = y_{t-s}
\]
where:
- \( s \) is the seasonal period (e.g., for monthly data, \( s = 12 \) for yearly seasonality).
- The model assumes that the pattern **repeats every season**.

### **Forecasting Process**  
1. Identify the seasonality period \( s \).
2. Take the value observed **one full season ago**.
3. Use it as the forecast.

#### **Example**
For monthly sales data where \( s = 12 \),  
If January 2023 sales were **200 units**,  
\[
\hat{y}_{\text{Jan 2024}} = y_{\text{Jan 2023}} = 200
\]

---

## **5. Window Average Model**  
Instead of averaging all past values, this model **takes only the most recent \( w \) values** into account.

### **Formula Breakdown**  
\[
\hat{y}_{t+1} = \frac{1}{w} \sum_{j=t-w+1}^{t} y_j
\]
where:
- \( w \) is the **window size** (number of recent observations to consider).
- The model averages only the most **recent \( w \) observations**.

### **Forecasting Process**  
1. Choose a window size \( w \).
2. Take the last \( w \) values and compute their mean.
3. Use this mean as the forecast.

#### **Example**
If the last **3** observations are \( [14, 16, 18] \) and \( w = 3 \),  
\[
\hat{y}_{t+1} = \frac{14 + 16 + 18}{3} = 16
\]

---

## **6. Seasonal Window Average Model**  
This model improves **Seasonal Naïve** by averaging multiple past seasonal values instead of using just one.

### **Formula Breakdown**  
\[
\hat{y}_{t+1} = \frac{1}{N} \sum_{k=1}^{N} y_{t - k \cdot s}
\]
where:
- \( N \) is the number of past seasons to consider.
- \( s \) is the seasonality period.
- The model **smooths out seasonal variations**.

### **Forecasting Process**  
1. Choose the number of past seasons \( N \).
2. Take the values from **previous seasons**.
3. Compute their average.
4. Use this average as the forecast.

#### **Example**
For quarterly sales data (\( s = 4 \)) with the last **2** seasonal values:  
If past Q1 sales were **[200, 220]**,  
\[
\hat{y}_{Q1, t+1} = \frac{200 + 220}{2} = 210
\]

---

## **Comparison of Baseline Models**

| Model                     | Captures Trends? | Captures Seasonality? | Complexity |
|---------------------------|-----------------|-----------------------|------------|
| **Historic Average**       | ❌ No           | ❌ No                  | ⭐ Simple  |
| **Naïve**                 | ❌ No           | ❌ No                  | ⭐ Simple  |
| **Random Walk with Drift** | ✅ Yes          | ❌ No                  | ⭐⭐ Medium |
| **Seasonal Naïve**         | ❌ No           | ✅ Yes                 | ⭐ Simple  |
| **Window Average**         | ✅ Partial      | ❌ No                  | ⭐⭐ Medium |
| **Seasonal Window Average** | ✅ Partial      | ✅ Yes                 | ⭐⭐ Medium |
