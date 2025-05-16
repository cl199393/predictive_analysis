## Install Packages
library("xgboost")
library(datasets)
library(caret)
library(tidyverse)
library("Ckmeans.1d.dp")
library(shapr)
library(dplyr)
library(pROC)
library(readr)
library(ggplot2)
library(reshape2)
library(corrplot)
if(!requireNamespace("lubridate", quietly = TRUE)) install.packages("lubridate")
if(!requireNamespace("rmarkdown", quietly = TRUE)) install.packages("rmarkdown")
if(!requireNamespace("mixsmsn", quietly = TRUE)) install.packages("mixsmsn")
if (!requireNamespace("GGally", quietly = TRUE)) install.packages("GGally")
if (!requireNamespace("car", quietly = TRUE)) install.packages("car")
if (!require(randomForest)) {
  install.packages("randomForest")
}
install.packages("e1071")
library(randomForest)
install.packages("ranger")
install.packages("iml")
install.packages("shapr")
library(shapr)
library(iml)
library(car)
library(lubridate)
library(rmarkdown)
library(mixsmsn)
library(GGally)
library(ranger)
library(e1071)
install.packages("gtsummary")
library(gtsummary)
library(gt)
install.packages("broom")
library(broom)
install.packages("grpreg")
library(grpreg)

## Load Data
data <- readRDS("/Users/changliu/Desktop/Big_Data/Research/Physics/Education_XGBoost/SP2023_Exam2/forML3.Rds")
write.csv(data, "/Users/changliu/Desktop/Big_Data/Research/Physics/Education_XGBoost/SP2023_Exam2/forML3.csv", row.names = FALSE)
data
summary(data)

## Data Process
data <- data %>%
  na.omit() %>%
  mutate(
    long_similar = as.factor(long_similar),
    long_identical = as.factor(long_identical),
    Q6_version = as.factor(Q6_version),
    is_similar = as.factor(is_similar),
    is_identical = as.factor(is_identical),
    Original_outcome = as.factor(Original_outcome), 
    Transfer_outcome = as.factor(Transfer_outcome),
    Total_outcome = as.factor(Total_outcome),
    med_lead_to_Exam = as.numeric(med_lead_to_Exam, units = "mins"),
    last_practice_to_Exam = as.numeric(last_practice_to_Exam, units = "mins")
  ) 

data <- data %>%
  rename(
    user_id = `user_id`,
    midterm_exam_3 = `Midterm Exam 3`,
    midterm_exam_2 = `Midterm Exam 2`,
    midterm_exam_score = `Midterm Exam Score`,
    final_exam = `Final Exam`,
    midterm_exam_1 = `Midterm Exam 1`,
    course_final_score = `Course Final Score`
  ) %>%
  dplyr::select(-c(user_id, midterm_exam_3, midterm_exam_2, midterm_exam_score, final_exam,Total_outcome,course_final_score))


### Split data
original_outcome_data <- data %>%
  dplyr::select(-c(Transfer_outcome))

transfer_outcome_data <- data %>%
  dplyr::select(-c(Original_outcome))


### Choose the original outcome data
summary(original_outcome_data)
summary(transfer_outcome_data)
# Create the summary table
table1 <- tbl_summary(
  data = original_outcome_data,
  by = Original_outcome, # Split table by cluster_id_numeric
  statistic = list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~ "{n} ({p}%)"),
  missing = "no"
) %>%
  add_n() %>% # Add column with total number of non-missing observations
  add_p() %>% # Test for a difference between groups
  modify_header(label = "**Variable**") %>% # Update the column header
  bold_labels() %>% # Bold labels
  as_gt() %>% # Convert to gt table for better formatting
  gt::tab_header(title = "Original Outcome Descriptive Summary Table") # Add a title to the table

table2 <- tbl_summary(
  data = transfer_outcome_data,
  by = Transfer_outcome, # Split table by cluster_id_numeric
  statistic = list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~ "{n} ({p}%)"),
  missing = "no"
) %>%
  add_n() %>% # Add column with total number of non-missing observations
  add_p() %>% # Test for a difference between groups
  modify_header(label = "**Variable**") %>% # Update the column header
  bold_labels() %>% # Bold labels
  as_gt() %>% # Convert to gt table for better formatting
  gt::tab_header(title = "Transfer Outcome Descriptive Summary Table")



# Print the table
table1
table2

gtsave(data = table1, filename = "Original_Outcome_Descriptive_Summary_Table.docx")
gtsave(data = table2, filename = "Transfer_Outcome_Descriptive_Summary_Table.docx")


## Check the data correlation
cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")
print(cor_matrix)

melted_cor_matrix <- melt(cor_matrix)

correlation <- ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed()
ggsave(filename = "correlation_plot.png", plot = correlation)


png(filename = "correlation_plot3.png",width = 4000, height = 3000, res = 300)
corrplot(cor_matrix, method = "circle")
dev.off()

# Perform hierarchical clustering on the correlation matrix
dist_matrix <- as.dist(1 - abs(cor_matrix))
hc <- hclust(dist_matrix)
# Plot the dendrogram
png(filename = "hierarchical_clustering_dendrogram.png", width = 1600, height = 1200, res = 300)
# Plot the dendrogram
plot(hc, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "", cex = 0.9)
# Close the PNG device
dev.off()

## histogram for continuous variables
create_histogram <- function(data, variable) {
  # Convert to numeric if the variable is a factor
  if (is.factor(data[[variable]])) {
    data[[variable]] <- as.numeric(as.character(data[[variable]]))
  }
  
  # Create histogram
  hist(data[[variable]], main = paste("Histogram of", variable), xlab = variable, 
       col = "lightblue", border = "black")
  
  # Save the histogram as a PNG file
  file_name <- paste0(variable, "_histogram.png")
  dev.copy(png, file_name)
  dev.off()
}

# Call the function to create and save histograms for the entire dataset
variables <- colnames(data)
for (variable in variables) {
  create_histogram(data, variable)
}
create_histogram(data, "frac_correct")
create_histogram(data, "midterm_exam_1")
create_histogram(data, "course_final_score")


## Transformation
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

standardize <- function(x) {
  return ((x - mean(x)) / sd(x))
}

original_outcome_data_transfer <- original_outcome_data %>%
  mutate(
    nPracticed_log = log(nPracticed+1),
    medTime_correct_log = log(medTime_correct+1),
    med_lead_to_Exam_log = log(med_lead_to_Exam/60+1),
    last_practice_to_Exam_log = log(last_practice_to_Exam/60+1),
    fracLong_log = log(fracLong+1),
    frac_correct_std = standardize(original_outcome_data$frac_correct)
  )


par(mfrow = c(2, 3))

hist(original_outcome_data_transfer$nPracticed_log, main = "Log-Transformed nPracticed", xlab = "log_nPracticed", col = "lightblue")
hist(original_outcome_data_transfer$fracLong_log, main = "Log-Transformed fracLong", xlab = "log_fracLong", col = "lightblue")
hist(original_outcome_data_transfer$med_lead_to_Exam_log, main = "Log-Transformed med_lead_to_Exam", xlab = "log_med_lead_to_Exam", col = "lightblue")
hist(original_outcome_data_transfer$last_practice_to_Exam_log, main = "Log-Transformed last_practice_to_Exam", xlab = "log_last_practice_to_Exam", col = "lightblue")
hist(original_outcome_data_transfer$medTime_correct_log, main = "Log-Transformed medTime_correct", xlab = "medTime_correct_log", col = "lightblue")
hist(original_outcome_data_transfer$frac_correct_std, main = "Standardise frac_correct", xlab = "frac_correct_std", col = "lightblue")

## transfer data
transfer_outcome_data_transfer <- transfer_outcome_data %>%
  mutate(
    nPracticed_log = log(nPracticed+1),
    medTime_correct_log = log(medTime_correct+1),
    med_lead_to_Exam_log = log(med_lead_to_Exam/60+1),
    last_practice_to_Exam_log = log(last_practice_to_Exam/60+1),
    fracLong_log = log(fracLong+1),
    frac_correct_std = standardize(frac_correct)
  )


par(mfrow = c(2, 3))

hist(transfer_outcome_data_transfer$nPracticed_log, main = "Log-Transformed nPracticed", xlab = "log_nPracticed", col = "lightblue")
hist(transfer_outcome_data_transfer$fracLong_log, main = "Log-Transformed fracLong", xlab = "log_fracLong", col = "lightblue")
hist(transfer_outcome_data_transfer$med_lead_to_Exam_log, main = "Log-Transformed med_lead_to_Exam", xlab = "log_med_lead_to_Exam", col = "lightblue")
hist(transfer_outcome_data_transfer$last_practice_to_Exam_log, main = "Log-Transformed last_practice_to_Exam", xlab = "log_last_practice_to_Exam", col = "lightblue")
hist(transfer_outcome_data_transfer$medTime_correct_log, main = "Log-Transformed medTime_correct", xlab = "medTime_correct_log", col = "lightblue")
hist(transfer_outcome_data_transfer$frac_correct_std, main = "Standardise frac_correct", xlab = "frac_correct_std", col = "lightblue")

### new dataframe
original_outcome_data_transfer <- original_outcome_data_transfer %>%
  dplyr::select(-c(nPracticed, medTime_correct, fracLong, med_lead_to_Exam, last_practice_to_Exam, frac_correct))

transfer_outcome_data_transfer <- transfer_outcome_data_transfer %>%
  dplyr::select(-c(nPracticed, medTime_correct, fracLong, med_lead_to_Exam, last_practice_to_Exam, frac_correct))

### Data Split for ML
set.seed(215)
trainIndex_original <- createDataPartition(original_outcome_data_transfer$Original_outcome, p = .8, 
                                           list = FALSE, 
                                           times = 1)
trainData_original <- original_outcome_data_transfer[ trainIndex_original,]
testData_original  <- original_outcome_data_transfer[-trainIndex_original,]
train_data_original <- trainData_original[, -which(names(trainData_original) == "Original_outcome")]
train_labels_original <- as.factor(trainData_original$Original_outcome)
test_data_original <- testData_original[, -which(names(testData_original) == "Original_outcome")]
test_labels_original <- as.factor(testData_original$Original_outcome)


trainIndex_transfer <- createDataPartition(transfer_outcome_data_transfer$Transfer_outcome, p = .8, 
                                           list = FALSE, 
                                           times = 1)
trainData_transfer <- transfer_outcome_data_transfer[ trainIndex_transfer,]
testData_transfer  <- transfer_outcome_data_transfer[-trainIndex_transfer,]
train_data_transfer <- trainData_transfer[, -which(names(trainData_transfer) == "Transfer_outcome")]
train_labels_transfer <- as.factor(trainData_transfer$Transfer_outcome)
test_data_transfer <- testData_transfer[, -which(names(testData_transfer) == "Transfer_outcome")]
test_labels_transfer <- as.factor(testData_transfer$Transfer_outcome)



#### SHAP Plot Function
create_shap_plot <- function(shap_df, title = "Feature Importance - Logistic Regression Model", 
                             x_title = "Feature", y_title = "SHAP Value") {
  # Ensure the dataframe has the expected columns
  if (!all(c("feature", "phi", "color") %in% colnames(shap_df))) {
    stop("The data frame must contain 'feature', 'phi', and 'color' columns")
  }
  
  # Create the plot
  shap_plot <- ggplot(shap_df, aes(x = reorder(feature, abs(phi)), y = phi, fill = color)) +
    geom_bar(stat = "identity") +
    scale_fill_identity() +
    coord_flip() + # This will flip the plot so the largest impact is at the top
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.title = element_text(size = 16, face = "bold"),
      axis.text = element_text(size = 14, face = "bold"),
      axis.title.x = element_text(size = 12, face = "bold"), #, hjust = -0.3, vjust = 3),  # Adjust the margins around the x-axis title
      axis.title.y = element_text(size = 12, face = "bold") #, hjust = 0, vjust = 6)
    ) +
    labs(title = title, x = x_title, y = y_title)+
    geom_text(aes(label = sprintf("%+0.3f", phi), y = phi, hjust = ifelse(phi > 0, 1.1, -0.1)),
              color = "black", size = 5, face = "bold")
  
  # Return the plot
  return(shap_plot)
}


## Fit Mode
#######__________Logisitic Regression________________________________________
# Make predictions
logistic_transfer <- glm(Transfer_outcome ~ ., data = trainData_transfer, family = binomial)
summary(logistic_transfer)
logistic_tidy_transfer <- tidy(logistic_transfer)
write.csv(logistic_tidy_transfer, file = "logistic_regression_results_transfer_nofinalexamscore.csv", row.names = FALSE)


## plot coefficient
logistic_tidy_transfer <- logistic_tidy_transfer %>%
  mutate(direction = ifelse(estimate > 0, "Positive", "Negative")) %>%
  filter(term != "(Intercept)")

losgistic_plot_coeff_transfer = ggplot(logistic_tidy_transfer, aes(x = reorder(term, abs(estimate)), y = estimate, fill = direction)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance -- Logistic Regression Coefficients",
       x = "Features",
       y = "Coefficient Estimate") +
  scale_fill_manual(values = c("Positive" = "#FF007F", "Negative" = "#5B9BD5")) +
  theme(axis.text = element_text(size = 12, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  geom_text(aes(label = sprintf("%+0.3f", estimate), y = estimate, hjust = ifelse(estimate > 0, 1.1, -0.1)),
            color = "black", size = 5, face = "bold")
ggsave(filename = "Feature_importance_Logistic_Regression_coefficient_transfer.png", losgistic_plot_coeff_transfer, dpi = 300)


# Make predictions
logistic_pred_transfer <- predict(logistic_transfer, newdata = testData_transfer)

logistic_pred_class_transfer <- ifelse(logistic_pred_transfer > 0.5, "1", "0")

# Evaluate the model
logistic_conf_matrix_transfer <- confusionMatrix(as.factor(logistic_pred_class_transfer), testData_transfer$Transfer_outcome)
print(logistic_conf_matrix_transfer)
# Extract the confusion matrix table
lr_table_transfer <- logistic_conf_matrix_transfer$table
# Calculate Precision, Recall, and F1 Score
precision_transfer <- lr_table_transfer[1, 1] / sum(lr_table_transfer[1, ])
recall_transfer <- lr_table_transfer[1, 1] / sum(lr_table_transfer[, 1])
f1_score_transfer <- 2 * (precision_transfer * recall_transfer) / (precision_transfer + recall_transfer)
# Print the results
cat("Precision: ", precision_transfer, "\n")
cat("Recall: ", recall_transfer, "\n")
cat("F1 Score: ", f1_score_transfer, "\n")
# ROC Plot
logistic_roc_transfer <- roc(as.numeric(testData_transfer$Transfer_outcome) - 1, logistic_pred_transfer)
# Plot ROC curve
plot(logistic_roc_transfer, main = "ROC Curve - Logistic Model - Transfer", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
logistic_auc_transfer <- auc(logistic_roc_transfer)
logistic_auc_ci_transfer <- ci.auc(logistic_roc_transfer)
# Print AUC and its 95% CI
cat("AUC:", logistic_auc_transfer, "\n")
cat("95% CI for AUC:", logistic_auc_ci_transfer, "\n")



# SHAP Value
# Define a custom prediction function for the logistic model
set.seed(123)
logistic_predict_transfer <- function(model, newdata) {
  as.numeric(predict(model, newdata, type = "response"))
}

# Create a Predictor object using the custom prediction function
logistic_shap_predictor_transfer <- Predictor$new(
  model = logistic_transfer,
  data = train_data_transfer,
  y = train_labels_transfer,
  predict.function = logistic_predict_transfer
)
# Check the predictor object to ensure it is correctly initialized
logistic_shap_predictor_transfer
# Use the Shapley method to compute SHAP values for a sample of test data
logistic_shapley_transfer <- Shapley$new(predictor = logistic_shap_predictor_transfer, x.interest = test_data_transfer)
# Get SHAP values
logistic_shap_values_transfer <- logistic_shapley_transfer$results
print(logistic_shap_values_transfer)
# Create a factor for color based on the sign of phi
logistic_shap_values_transfer$color <- ifelse(logistic_shap_values_transfer$phi > 0, '#FF007F', '#5B9BD5')
#logistic_importance_df$feature <- factor(logistic_importance_df$feature, levels = rev(logistic_importance_df$feature))
logistic_importance_df_transfer <- data.frame(feature = logistic_shap_values_transfer$feature, phi = logistic_shap_values_transfer$phi, color = logistic_shap_values_transfer$color)
logistic_importance_df_transfer
# Sort the dataframe by the absolute values of the coefficients in descending order
logistic_importance_df_transfer <- logistic_importance_df_transfer[order(-logistic_shap_values_transfer$phi), ]


# Now create the plot with the newly ordered factors
logistic_plot_transfer <- create_shap_plot(logistic_importance_df_transfer, title = "Feature Importance - Logistic Regression Model - Transfer")
# Display the plot
print(logistic_plot_transfer)
ggsave(filename = "Feature_importance_Logistic_Regression_transfer.png", logistic_plot_transfer, dpi = 300)

## Top 5
# Select the top 5 features based on the absolute values of the coefficients
logistic_importance_df_top5_transfer <- logistic_importance_df_transfer%>%
  arrange(desc(abs(phi)))%>%
  head(5)
# Now create the plot with the newly ordered factors -- Top 5
logistic_plot_top5_transfer <- create_shap_plot(logistic_importance_df_top5_transfer, title = "Top 5 Feature Importance - Logistic Regression Model - Transfer", 
                                       x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(logistic_plot_top5_transfer)
ggsave(filename = "Top5_Feature_importance_Logistic_Regression_transfer.png", logistic_plot_top5_transfer, dpi = 300)

#####__________________________Random Forest________________________________
levels(trainData_transfer$Transfer_outcome) <- make.names(levels(trainData_transfer$Transfer_outcome))
levels(testData_transfer$Transfer_outcome) <- make.names(levels(testData_transfer$Transfer_outcome))
trainData_transfer$Transfer_outcome <- factor(trainData_transfer$Transfer_outcome, levels = levels(trainData_transfer$Transfer_outcome))
testData_transfer$Transfer_outcome <- factor(testData_transfer$Transfer_outcome, levels = levels(testData_transfer$Transfer_outcome))
# Define the parameter grid for hyperparameter tuning
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 5, 10)
)
# Set up cross-validation
rf_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)
# Train the model with hyperparameter tuning and calculate importance
rf_train_transfer <- train(
  Transfer_outcome ~ .,
  data = trainData_transfer,
  method = "ranger",
  trControl = rf_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  importance = 'impurity'  # Ensure importance is calculated
)
# Print the best model
print(rf_train_transfer)
# fit the best model
rf_model_transfer <- randomForest(
  Transfer_outcome ~ .,
  data = trainData_transfer,
  mtry = rf_train_transfer$bestTune$mtry,
  nodesize = rf_train_transfer$bestTune$min.node.size,
  importance = TRUE
)
summary(rf_model_transfer)
# Make predictions with the best model
rf_pred_transfer <- predict(rf_model_transfer, newdata = testData_transfer, type = "prob")
rf_pred_classes_transfer <- ifelse(rf_pred_transfer[, "X1"] > 0.5, "X1", "X0")
# Evaluate the model
rf_conf_matrix_transfer <- confusionMatrix(as.factor(rf_pred_classes_transfer), testData_transfer$Transfer_outcome)
print(rf_conf_matrix_transfer)
# Extract the confusion matrix table
rf_table_transfer <- rf_conf_matrix_transfer$table
# Calculate Precision, Recall, and F1 Score
precision_transfer <- rf_table_transfer[1, 1] / sum(rf_table_transfer[1, ])
recall_transfer <- rf_table_transfer[1, 1] / sum(rf_table_transfer[, 1])
f1_score_transfer <- 2 * (precision_transfer * recall_transfer) / (precision_transfer + recall_transfer)
# Print the results
cat("Precision: ", precision_transfer, "\n")
cat("Recall: ", recall_transfer, "\n")
cat("F1 Score: ", f1_score_transfer, "\n")
# Create ROC curve
rf_roc_transfer<- roc(as.numeric(testData_transfer$Transfer_outcome) - 1, rf_pred_transfer[,2])
# Plot ROC curve
plot(rf_roc_transfer, main = "ROC Curve - Random Forest Model (Tuned)", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
rf_auc_transfer <- auc(rf_roc_transfer)
rf_auc_ci_transfer <- ci.auc(rf_auc_transfer)
# Print AUC and its 95% CI
cat("AUC:", rf_auc_transfer, "\n")
cat("95% CI for AUC:", rf_auc_ci_transfer, "\n")

# Feature importance
set.seed(123)
# Define a custom prediction function for the Random Forest model
rf_predict_transfer <- function(model, newdata) {
  predict(model, newdata, type = "prob")[, 2]
}
rf_shap_predictor_transfer <- Predictor$new(
  model = rf_model_transfer,
  data = train_data_transfer,
  y = train_labels_transfer,
  predict.function = rf_predict_transfer
)
# Use the Shapley method to compute SHAP values for a sample of test data
rf_shapley_transfer <- Shapley$new(predictor = rf_shap_predictor_transfer, x.interest = test_data_transfer)
# Get SHAP values
rf_shap_values_transfer <- rf_shapley_transfer$results
print(rf_shap_values_transfer)
# Create a factor for color based on the sign of phi
rf_shap_values_transfer$color <- ifelse(rf_shap_values_transfer$phi > 0, '#FF007F', '#5B9BD5')
rf_importance_df_transfer <- data.frame(feature = rf_shap_values_transfer$feature, phi = rf_shap_values_transfer$phi, color = rf_shap_values_transfer$color)
rf_importance_df_transfer
# Sort the dataframe by the absolute values of the coefficients in descending order
rf_importance_df_transfer <- rf_importance_df_transfer[order(-rf_shap_values_transfer$phi), ]

# Now create the plot with the newly ordered factors
rf_plot_transfer <- create_shap_plot(rf_importance_df_transfer, title = "Feature Importance - Random Forest Model - Transfer", 
                            x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(rf_plot_transfer)
ggsave(filename = "Feature_importance_Random_Forest_transfer.png", rf_plot_transfer, dpi = 300)
## Top 5
rf_importance_df_top5_transfer <- rf_importance_df_transfer%>%
  arrange(desc(abs(phi)))%>%
  head(5)
# Now create the plot with the newly ordered factors -- Top 5
rf_plot_top5_transfer <- create_shap_plot(rf_importance_df_top5_transfer, title = "Top 5 Feature Importance - Random Forest Model - Transfer", 
                                 x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(rf_plot_top5_transfer)
ggsave(filename = "Top5_Feature_importance_Random_Forest_transfer.png", rf_plot_top5_transfer, dpi = 300)

########_______________________XGBoost______________________________
trainData_transfer <- trainData_transfer %>%
  mutate(across(where(is.factor), as.numeric))
testData_transfer <- testData_transfer %>%
  mutate(across(where(is.factor), as.numeric))
train_data_transfer <- train_data_transfer %>%
  mutate(across(where(is.factor), as.numeric))
test_data_transfer <- test_data_transfer %>%
  mutate(across(where(is.factor), as.numeric))
train_matrix_transfer <- xgb.DMatrix(data = as.matrix(trainData_transfer %>% dplyr::select(-Transfer_outcome)), label = as.numeric(trainData_transfer$Transfer_outcome) - 1)
test_matrix_transfer <- xgb.DMatrix(data = as.matrix(testData_transfer %>% dplyr::select(-Transfer_outcome)), label = as.numeric(testData_transfer$Transfer_outcome) - 1)
dtrain_transfer <- xgb.DMatrix(data = as.matrix(train_data_transfer), label = train_labels_transfer)
dtest_transfer <- xgb.DMatrix(data = as.matrix(test_data_transfer), label = test_labels_transfer)
trainData_transfer$Transfer_outcome <- as.factor(trainData_transfer$Transfer_outcome)

# Define the parameters for the xgboost model
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.3,
  nthread = 2,
  verbosity = 1
)
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1, 5),
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.6, 0.8, 1)
)
xgb_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# Convert levels of Transfer_outcome to valid R variable names
levels(trainData_transfer$Transfer_outcome) <- make.names(levels(trainData_transfer$Transfer_outcome))

# Train the xgboost model
xgb_train_transfer <- train(
  Transfer_outcome ~ .,
  data = trainData_transfer,
  method = "xgbTree",
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  metric = "ROC"
)
#summary(xgb_model)
print(xgb_train_transfer)
# Fit model
set.seed(123)
params_transfer <- list(
  objective = "binary:logistic",
  max_depth = xgb_train_transfer$bestTune$max_depth,
  eta = xgb_train_transfer$bestTune$eta,
  gamma = xgb_train_transfer$bestTune$gamma,
  colsample_bytree = xgb_train_transfer$bestTune$colsample_bytree,
  min_child_weight = xgb_train_transfer$bestTune$min_child_weight,
  subsample = xgb_train_transfer$bestTune$subsample
)
# Set the number of rounds (iterations)
nrounds_transfer <- xgb_train_transfer$bestTune$nrounds

# Train the model
xgb_model_transfer <- xgb.train(
  params = params_transfer,
  data = train_matrix_transfer,
  nrounds = nrounds_transfer,
  watchlist = list(train = train_matrix_transfer, test = test_matrix_transfer),
  eval_metric = "auc",
  verbose = 1
)
summary(xgb_model_transfer)
# Make predictions
#xgb_pred <- predict(xgb_model, newdata = test_matrix)
xgb_pred_transfer <- predict(xgb_model_transfer, newdata = test_matrix_transfer, type = "prob")
xgb_pred_class_transfer<- ifelse(xgb_pred_transfer> 0.5, "X1", "X0")
trainData_transfer$Transfer_outcome <- as.factor(trainData_transfer$Transfer_outcome)
testData_transfer$Transfer_outcome <- as.factor(testData_transfer$Transfer_outcome)
# Ensure that the predicted classes are factors
xgb_pred_class_transfer <- as.factor(xgb_pred_class_transfer)
# Match levels between the predicted classes and true classes
levels(xgb_pred_class_transfer) <- levels(testData_transfer$Transfer_outcome)
# Create the confusion matrix
xgb_conf_matrix_transfer <- confusionMatrix(xgb_pred_class_transfer, testData_transfer$Transfer_outcome)
# Print the confusion matrix
print(xgb_conf_matrix_transfer)
# Extract the confusion matrix table
xgb_table_transfer <- xgb_conf_matrix_transfer$table
# Calculate Precision, Recall, and F1 Score
precision_transfer <- xgb_table_transfer[1, 1] / sum(xgb_table_transfer[1, ])
recall_transfer <- xgb_table_transfer[1, 1] / sum(xgb_table_transfer[, 1])
f1_score_transfer <- 2 * (precision_transfer * recall_transfer) / (precision_transfer + recall_transfer)
# Print the results
cat("Precision: ", precision_transfer, "\n")
cat("Recall: ", recall_transfer, "\n")
cat("F1 Score: ", f1_score_transfer, "\n")
# ROC
xgb_roc_transfer <- roc(as.numeric(testData_transfer$Transfer_outcome) - 1, xgb_pred_transfer)
# Plot ROC curve
plot(xgb_roc_transfer, main = "ROC Curve - XGBoost Model - Transfer", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
xgb_auc_transfer <-auc(xgb_roc_transfer)
xgb_auc_ci_transfer <- ci.auc(xgb_roc_transfer)
# Print AUC and its 95% CI
cat("AUC:", xgb_auc_transfer, "\n")
cat("95% CI for AUC:", xgb_auc_ci_transfer, "\n")

# Feature importance
set.seed(123)
# Define a custom prediction function for the XGBoost model
predict_xgb_transfer <- function(model, newdata) {
  predict(model, newdata = xgb.DMatrix(data = as.matrix(newdata)))
}
# Create a Predictor object for the iml package
xgb_predictor_transfer <- Predictor$new(
  model = xgb_model_transfer,
  data = as.data.frame(train_data_transfer),  # Ensure this is a data frame
  y = train_labels_transfer,
  predict.function = predict_xgb_transfer
)
subset_test_data_transfer <- as.data.frame(test_data_transfer)
# Calculate SHAP values using the iml package
xgb_shapley_transfer <- Shapley$new(xgb_predictor_transfer, x.interest = subset_test_data_transfer)
# Get SHAP values
xgb_shap_values_transfer <- xgb_shapley_transfer$results
print(xgb_shap_values_transfer)
# Create a factor for color based on the sign of phi
xgb_shap_values_transfer$color <- ifelse(xgb_shap_values_transfer$phi > 0, '#FF007F', '#5B9BD5')
xgb_importance_df_transfer <- data.frame(feature = xgb_shap_values_transfer$feature, phi = xgb_shap_values_transfer$phi, color = xgb_shap_values_transfer$color)
xgb_importance_df_transfer
# Sort the dataframe by the absolute values of the coefficients in descending order
xgb_importance_df_transfer <- xgb_importance_df_transfer[order(-xgb_shap_values_transfer$phi), ]
# Now create the plot with the newly ordered factors
xgb_plot_transfer <- create_shap_plot(xgb_importance_df_transfer, title = "Feature Importance - XGboost Model - Transfer", 
                             x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(xgb_plot_transfer)
ggsave(filename = "Feature_importance_XGBoost_transfer.png", xgb_plot_transfer, dpi = 300)
## Top 5 
xgb_importance_df_top5_transfer <- xgb_importance_df_transfer %>%
  arrange(desc(abs(phi))) %>%
  head(5)
# Create the plot
# Now create the plot with the newly ordered factors
xgb_plot_top5_transfer <- create_shap_plot(xgb_importance_df_top5_transfer, title = "Top 5 Feature Importance - XGboost Model - Transfer", 
                                  x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(xgb_plot_top5_transfer)
ggsave(filename = "Top5_Feature_importance_XGBoost_transfer.png", xgb_plot_top5_transfer, dpi = 300)


########__________________________Group Lasso________________________________________-
cor_matrix_group_lasso_transfer <- cor(train_data_transfer, use = "complete.obs")
# Perform hierarchical clustering on the correlation matrix
dist_matrix_transfer <- as.dist(1 - abs(cor_matrix_group_lasso_transfer))
hc_transfer <- hclust(dist_matrix_transfer)
# Cut the dendrogram to form clusters
group_cutoff <- 0.5  # Adjust this value as needed
groups_transfer <- cutree(hc_transfer, h = group_cutoff)
# Convert Original_outcome to numeric
trainData_transfer$Transfer_outcome <- as.numeric(as.factor(trainData_transfer$Transfer_outcome)) - 1
# Prepare the predictor matrix and response variable
X_transfer <- as.matrix(train_data_transfer)
y_transfer <- trainData_transfer$Transfer_outcome
# Fit the group lasso model
group_lasso_transfer <- grpreg(X_transfer, y_transfer, group = groups_transfer,family = "binomial")
# Print the fit summary
summary(group_lasso_transfer)
# Perform cross-validation to select the best lambda
cv_group_lasso_transfer <- cv.grpreg(X_transfer, y_transfer, group = groups_transfer, family = "binomial")
# Plot the cross-validation curve
png(filename = "Group_Selection_transfer.png", width = 1600, height = 1200, res = 300)
plot(cv_group_lasso_transfer)
dev.off()
# Get the best lambda value
best_lambda_transfer <- cv_group_lasso_transfer$lambda.min
# Fit the model with the best lambda
group_lasso_model_transfer <- grpreg(X_transfer, y_transfer, group = groups_transfer, family = "binomial", lambda = best_lambda_transfer)
# Print the final model summary
summary(group_lasso_model_transfer)

# Make predictions
X_test_transfer<- as.matrix(test_data_transfer)
group_lasso_pred_transfer <- predict(group_lasso_model_transfer, X = X_test_transfer, type = "response")
group_lasso_pred_class_transfer <- ifelse(group_lasso_pred_transfer> 0.5, "X1", "X0")
# Ensure that the predicted classes are factors
group_lasso_pred_class_transfer <- as.factor(group_lasso_pred_class_transfer)
# Match levels between the predicted classes and true classes
levels(group_lasso_pred_class_transfer) <- levels(testData_transfer$Transfer_outcome)
# Create the confusion matrix
group_lasso_conf_matrix_transfer <- confusionMatrix(group_lasso_pred_class_transfer, testData_transfer$Transfer_outcome)
# Print the confusion matrix
print(group_lasso_conf_matrix_transfer)

# Extract the confusion matrix table
grop_lasso_table_transfer <- group_lasso_conf_matrix_transfer$table
# Calculate Precision, Recall, and F1 Score
precision_transfer <- grop_lasso_table_transfer[1, 1] / sum(grop_lasso_table_transfer[1, ])
recall_transfer <- grop_lasso_table_transfer[1, 1] / sum(grop_lasso_table_transfer[, 1])
f1_score_transfer <- 2 * (precision_transfer * recall_transfer) / (precision_transfer + recall_transfer)
# Print the results
cat("Precision: ", precision_transfer, "\n")
cat("Recall: ", recall_transfer, "\n")
cat("F1 Score: ", f1_score_transfer, "\n")
# ROC
group_lasso_roc_transfer <- roc(as.numeric(testData_transfer$Transfer_outcome) - 1, group_lasso_pred_transfer)
# Plot ROC curve
plot(group_lasso_roc_transfer, main = "ROC Curve - XGBoost Model - Transfer", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
group_lasso_auc_transfer <-auc(group_lasso_roc_transfer)
group_lasso_auc_ci_transfer <- ci.auc(group_lasso_roc_transfer)
# Print AUC and its 95% CI
cat("AUC:", group_lasso_auc_transfer, "\n")
cat("95% CI for AUC:", group_lasso_auc_ci_transfer, "\n")

# Feature importance
set.seed(123)
# Define a custom prediction function for the XGBoost model
predict_group_lasso_transfer <- function(group_lasso_model_transfer, newdata = X_test_transfer) {
  predict(group_lasso_model_transfer, X = X_test_transfer, type = "response")
}
# Create a Predictor object for the iml package
group_lasso_predictor_transfer <- Predictor$new(
  model = group_lasso_model_transfer,
  data = as.data.frame(X_test_transfer),  # Ensure this is a data frame
  y = testData_transfer$Transfer_outcome,
  predict.function = predict_group_lasso_transfer
)

# Convert to matrix
test_matrix_group_lasso_transfer <- as.matrix(X_test_transfer)
subset_test_data_transfer <- as.data.frame(test_matrix_group_lasso_transfer)
# Calculate SHAP values using the iml package
group_lasso_shapley_transfer <- Shapley$new(group_lasso_predictor_transfer, x.interest = subset_test_data_transfer)
# Get SHAP values
group_lasso_shap_values_transfer <- group_lasso_shapley_transfer$results
print(group_lasso_shap_values_transfer)
# Create a factor for color based on the sign of phi
group_lasso_shap_values_transfer$color <- ifelse(group_lasso_shap_values_transfer$phi > 0, '#FF007F', '#5B9BD5')
group_lasso_importance_df_transfer <- data.frame(feature = group_lasso_shap_values_transfer$feature, phi = group_lasso_shap_values_transfer$phi, color = group_lasso_shap_values_transfer$color)
group_lasso_importance_df_transfer
# Sort the dataframe by the absolute values of the coefficients in descending order
group_lasso_importance_df_transfer <- group_lasso_importance_df_transfer[order(-group_lasso_shap_values_transfer$phi), ]
# Now create the plot with the newly ordered factors
group_lasso_plot_transfer <- create_shap_plot(group_lasso_importance_df_transfer, title = "Feature Importance - Group Lasso Model - Transfer", 
                                     x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(group_lasso_plot_transfer)
ggsave(filename = "Feature_importance_Group_Lasso_transfer.png", group_lasso_plot_transfer, dpi = 300)
## Top 5 
group_lasso_importance_df_top5_transfer <- group_lasso_importance_df_transfer %>%
  arrange(desc(abs(phi))) %>%
  head(5)
# Create the plot
# Now create the plot with the newly ordered factors
group_lasso_plot_top5_transfer <- create_shap_plot(group_lasso_importance_df_top5_transfer, title = "Top 5 Feature Importance - Group Lasso Model - Transfer", 
                                          x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(group_lasso_plot_top5_transfer)
ggsave(filename = "Top5_Feature_importance_Group_Lasso_transfer.png", group_lasso_plot_top5_transfer, dpi = 300)

####——————————————————————-CART_______________________________________________-
set.seed(215)
# Define control parameters for cross-validation
train_control <- trainControl(
  method = "cv",    # Use k-fold cross-validation
  number = 5,       # Number of folds
  search = "grid"   # Grid search for hyperparameter tuning
)

# Define a grid of hyperparameters to tune
# 'cp' stands for complexity parameter; it controls the size of the decision tree
tune_grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Train the CART model using caret with hyperparameter tuning
cart_model_tuned <- train(
  Transfer_outcome ~ .,              # Formula for the model
  data = trainData_transfer,        # Training data
  method = "rpart",         # Specify CART (rpart) model
  trControl = train_control, # Cross-validation settings
  tuneGrid = tune_grid      # Hyperparameter grid
)

# Print the best model and its parameters
print(cart_model_tuned)
print(cart_model_tuned$bestTune)
# Visualize the best decision tree
rpart.plot(cart_model_tuned$finalModel, type = 3, extra = 101, fallen.leaves = TRUE, cex = 0.8, main = "Tuned CART Decision Tree")

# Predict on the testing set using the tuned model
predictions_transfer <- predict(cart_model_tuned, newdata = testData_transfer)

# Create a confusion matrix to evaluate the model performance
cart_confusion_matrix_transfer <- table(Predicted = predictions_transfer, Actual = testData_transfer$Transfer_outcome)
print(cart_confusion_matrix_transfer)

# Calculate Precision, Recall, and F1 Score
precision_transfer <- cart_confusion_matrix_transfer[1, 1] / sum(cart_confusion_matrix_transfer[1, ])
recall_transfer <- cart_confusion_matrix_transfer[1, 1] / sum(cart_confusion_matrix_transfer[, 1])
f1_score_transfer <- 2 * (precision_transfer * recall_transfer) / (precision_transfer + recall_transfer)

# Print the results
cat("Precision: ", precision_transfer, "\n")
cat("Recall: ", recall_transfer, "\n")
cat("F1 Score: ", f1_score_transfer, "\n")

# Calculate ROC and AUC
# Make sure your 'cart_predictions' are probabilities for ROC calculation
predictions_prob_transfer <- predict(cart_model_tuned, newdata = testData_transfer, type = "prob")[, 2]  # Probabilities for the positive class

cart_roc_transfer <- roc(as.numeric(testData_transfer$Transfer_outcome) - 1, predictions_prob_transfer)  # Ensure 'cart_probabilities' are probabilities, not class labels

# Plot ROC curve
plot(cart_roc_transfer, main = "ROC Curve - CART Model", col = "#1c61b6")

# Calculate AUC (Area Under the Curve)
cart_auc_transfer <- auc(cart_roc_transfer)
cart_auc_ci_transfer <- ci.auc(cart_roc_transfer)

# Print AUC and its 95% Confidence Interval (CI)
cat("AUC:", cart_auc_transfer, "\n")
cat("95% CI for AUC:", cart_auc_ci_transfer, "\n")



#####_______________-ROC Plot_________________________________________
png(filename = "roc_curves_new_transfer_onlyexam1_2decimal.png", width = 1600, height = 1200, res = 300)
# Create a list of ROC curve objects with known names
roc_list_transfer <- list(
  CART = cart_roc_transfer,
  XGB = xgb_roc_transfer,
  RF = rf_roc_transfer
)

# Create a vector of model names
model_names_transfer <- names(roc_list_transfer)
curve_colors_transfer <- c("black", "#1E90FF", "#FF4D4F")  # Black for CART, Blue for XGB, and Red for RF

# Define a vector of colors for the ROC curves
#curve_colors_onlyexam1 <- rainbow(length(model_names_onlyexam1))

# Set the specific colors for each model
#curve_colors_onlyexam1[model_names_onlyexam1 == "Logistic"] <- "black"
#curve_colors_onlyexam1[model_names_onlyexam1 == "XGB"] <- "#1E90FF"
#curve_colors_onlyexam1[model_names_onlyexam1 == "RF"] <- "#FF4D4F"

# Plot the ROC curves without displaying AUC values
plot.roc(
  roc_list_transfer[[1]], 
  main = "ROC Curves", 
  col = curve_colors_transfer[1],  # Use black for Cart
  lwd = 4
)

# Plot the remaining ROC curves with the same line width
for (i in 2:length(model_names_transfer)) {
  plot.roc(
    roc_list[[i]],
    add = TRUE, 
    col = curve_colors_transfer[i],  # Use specified color for each model
    lwd = 4  # Set line width to 4
  )
}
# Add AUC values to the plot
auc_values_transfer <- sapply(roc_list_transfer, function(roc) auc(roc))

# Add AUC text to the plot
custom_auc_values_tranfer <- c("0.59", "0.66", "0.58")  # Replace with your desired AUC values
legend_text_transfer <- paste(model_names_transfer, "AUC:", custom_auc_values_tranfer)
legend("bottomright", legend = legend_text_transfer, col = curve_colors_transfer, lwd = 4, cex = 1)
# Close the PNG device
dev.off()

install.packages("magick")
library(magick)

img <- image_read("roc_curves_original_new.png")

svg_img <- image_convert(img, format = "svg")

image_write(svg_img, path = "roc_curves_original_new.svg", format = "svg")



saveRDS(cart_roc_transfer, "cart_transfer_auc.RDS", row.names = FALSE)




