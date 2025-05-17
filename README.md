# Towards Actionable Recommendations for Exam Preparation Using Isomorphic Problem Banks and Explanatory Machine Learning
 
The goal of this project is to enhance exam preparation strategies by leveraging isomorphic problem banks and explainable machine learning techniques.

## Code Path
These R scripts analyze feature importance from machine learning models used in the project.
### 1. Run predictive analysis:
```R
# Run feature importance analysis for the Original Problem
source("code/Feature_Importance.R")

# Run feature importance analysis for the Transfer Problem
source("code/Feature_Importance_transfer.R")

```

### 2. Run SHAP analysis for the best performance model:
This Jupyter notebook applies SHAP (SHapley Additive exPlanations) to visualize and interpret the most influential features in the best-performing model.
```Python
jupyter notebook code/SHAP.ipynb
```
