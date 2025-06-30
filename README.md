# ML-PayPredictor
analyzing salary through a lens — data-driven vision

## Goal
To analyze data science salary trends and predict salaries using machine learning, providing insights for job seekers and employers based on role, experience, company size, and working conditions.

---

## Background
The global demand for data professionals continues to surge as industries embrace AI, analytics, and digital transformation. Salary benchmarks play a crucial role in talent acquisition, retention, and strategic career planning.

This project uses a real-world dataset with over 59,000 entries (2020–2025), detailing job titles, experience levels, company size, employment type, and salary data. The aim is to uncover key salary trends and build a machine learning model (Random Forest Regressor) to predict salaries accurately.

The dataset was thoroughly cleaned, transformed, and engineered. This includes removing outliers, one-hot encoding categorical variables, frequency encoding job titles, and creating new features like `is_fully_remote` and `is_domestic`.

---

## Insights Summary

### Salary by Experience
- Entry-level (EN) salaries are generally the lowest, peaking around $90,000.  
- Senior-level (SE) and Expert (EX) professionals dominate higher salary brackets.  
- Mid-level (MI) roles show wide variation depending on job title and location.

### Remote Work Impact
- Fully remote roles are often associated with higher pay.  
- Remote jobs are increasingly common post-2021.  
- Domestic remote workers are more common than international ones.

### Geography Matters
- The US and select European countries dominate high-paying roles.  
- Salary disparity exists even among countries with similar cost of living.  
- Most companies prefer hiring domestically, affecting global talent dynamics.

### Job Title Frequency and Pay
- Machine Learning Engineers and Data Scientists are highly paid and frequently hired.  
- Rare job titles (e.g., AI Architect) skew toward higher salaries.  
- High job title frequency doesn’t always correlate with top-tier pay.

---

## Recommendations

- **For Job Seekers**: Target roles with growing demand and high median salaries like ML Engineer or AI Architect. Tailor your resume to match experience levels that fit higher-paying brackets.

- **For Recruiters**: Offer remote flexibility to stay competitive. Data shows fully remote jobs attract stronger candidates and are often linked to higher compensation.

- **For Policy Makers**: Address geographic disparities in tech compensation. Encourage local companies to adopt inclusive, borderless hiring practices where possible.

- **For New Graduates**: Entry-level roles vary in pay — consider internships or associate positions in countries or companies that pay above the median, even at entry levels.

---

## Model Results and Summary

The final model was a Random Forest Regressor trained after extensive preprocessing and feature engineering. A log transformation was applied to the salary target variable to improve prediction stability.

- **Mean Absolute Error (MAE)**: ~46,047  
- **Mean Squared Error (MSE)**: ~3.58 billion  
- **R² Score**: ~0.23

While the model doesn’t explain all salary variation, it captures several key patterns. Improvements are possible with more granular data like education level, company reputation, or job function specifics.

---

## Data Preprocessing Steps
- Dropped unnecessary columns (`salary`, `salary_currency`)
- Converted string types
- Standardized values (e.g., currency)
- Filtered rows for USD only
- Removed duplicates and outliers
- One-hot encoded `experience_level`, `employment_type`, `company_size`
- Frequency encoded `job_title`
- Engineered features: `is_fully_remote`, `is_domestic`
- Applied log transformation to `salary_in_usd`

---

## Visualizations
- Salary distribution by experience level
- Boxplots to detect outliers
- Heatmaps to analyze feature correlation
- Country-wise salary comparisons
- Job title frequency vs. pay

---

## Challenges Faced
- High cardinality in `job_title` required frequency encoding
- Skewed salary distribution handled using log transformation
- Outlier handling dropped ~2,900 rows
- Low R² score indicates the dataset has high variance not captured by current features

---

## Technologies Used
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter / Kaggle Notebook

---

## Future Work
- Try advanced regressors: XGBoost, LightGBM
- Incorporate external data: cost of living, education, company tier
- Build a frontend or dashboard (Streamlit or Gradio)
- Add NLP embeddings from job descriptions for deeper insights

---

## Author
**Pakkapol Satthapiti**  
MSC of Data Science and AI | The University of Liverpool 
*Feel free to connect!*

