# Melbourne Housing Market Analysis using Linear Regression

## Objective
To investigate the factors influencing housing prices in Melbourne using statistical modeling, focusing primarily on the effect of property **distance from the central business district (CBD)**, along with **building attributes** and **regional differences**.

## Approach
1. **Data Preprocessing**
   - Loaded and cleaned the **Melbourne housing dataset**.
   - Filtered the data to focus on **houses**, excluding units and townhouses.
   - Removed outliers such as properties located more than 20 km from the CBD.

2. **Feature Engineering**
   - Computed the natural logarithm of house prices (`logPrice`) to normalize skewed price distributions.
   - Created **dummy variables** for categorical features such as property **Regionname**.

3. **Modeling**
   - Developed several **linear regression models**:
     - Simple Model: `log(Price) ~ Distance`
     - Extended Model: Added predictors such as `BuildingArea`, `Rooms`, and regional dummy variables.
   - Models were created separately for:
     - Houses
     - Units
     - Townhouses
     - Combined property types

4. **Evaluation**
   - Used **adjusted R-squared** to evaluate model performance.
   - Compared different models to identify which configuration best explains price variations.

## Tools & Libraries Used
- **Python**
- **Pandas** for data manipulation  
- **NumPy** for numerical operations  
- **Matplotlib & Seaborn** for data visualization  
- **Statsmodels** for regression modeling and statistical summaries

## Key Insights
- **Distance to CBD** is inversely related to house prices: properties closer to the city tend to be more expensive.
- **Building attributes** like number of rooms and building area significantly improve model accuracy.
- **Regional dummy variables** enhance the model’s explanatory power by capturing location-specific effects.
- Among property types, **extended models for houses** yielded better adjusted R-squared scores, indicating they benefit the most from added predictors.
- The **combined extended model** generally performed well but not better than the best individual models for certain property types.
