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
install.packages("rpart")     # Install rpart package if not already installed
install.packages("rpart.plot") # Install rpart.plot package for visualization

library(rpart)
library(rpart.plot)

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
  dplyr::select(-c(user_id, midterm_exam_3, midterm_exam_2, midterm_exam_score, final_exam,Total_outcome))


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

table3 <- tbl_summary(
  data = original_outcome_data_transfer,
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
table3

## transfer data
transfer_outcome_data_transfer <- transfer_outcome_data %>%
  mutate(
    nPracticed_log = log(nPracticed+1),
    medTime_correct_log = log(medTime_correct+1),
    med_lead_to_Exam_log = log(med_lead_to_Exam/60+1),
    last_practice_to_Exam_log = log(last_practice_to_Exam/60+1),
    fracLong_log = log(fracLong+1),
    frac_correct_std = standardize(original_outcome_data$frac_correct)
  )


par(mfrow = c(2, 3))

hist(transfer_outcome_data_transfer$nPracticed_log, main = "Log-Transformed nPracticed", xlab = "log_nPracticed", col = "lightblue")
hist(transfer_outcome_data_transfer$fracLong_log, main = "Log-Transformed fracLong", xlab = "log_fracLong", col = "lightblue")
hist(transfer_outcome_data_transfer$med_lead_to_Exam_log, main = "Log-Transformed med_lead_to_Exam", xlab = "log_med_lead_to_Exam", col = "lightblue")
hist(transfer_outcome_data_transfer$last_practice_to_Exam_log, main = "Log-Transformed last_practice_to_Exam", xlab = "log_last_practice_to_Exam", col = "lightblue")
hist(transfer_outcome_data_transfer$medTime_correct_log, main = "Log-Transformed medTime_correct", xlab = "medTime_correct_log", col = "lightblue")
hist(transfer_outcome_data_transfer$frac_correct_std, main = "Standardise frac_correct", xlab = "frac_correct_std", col = "lightblue")


table4 <- tbl_summary(
  data = transfer_outcome_data_transfer,
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
table4




### new dataframe
original_outcome_data_transfer <- original_outcome_data_transfer %>%
  dplyr::select(-c(nPracticed, medTime_correct, fracLong, med_lead_to_Exam, last_practice_to_Exam, frac_correct))

original_outcome_data_transfer_onlyexam1 <- original_outcome_data_transfer %>%
  dplyr::select(-c(course_final_score))




transfer_outcome_data_transfer <- transfer_outcome_data_transfer %>%
  dplyr::select(-c(nPracticed, medTime_correct, fracLong, med_lead_to_Exam, last_practice_to_Exam, frac_correct))

### Data Split for ML
set.seed(215)
trainIndex <- createDataPartition(original_outcome_data_transfer$Original_outcome, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData <- original_outcome_data_transfer[ trainIndex,]
testData  <- original_outcome_data_transfer[-trainIndex,]
train_data <- trainData[, -which(names(trainData) == "Original_outcome")]
train_labels <- as.factor(trainData$Original_outcome)
test_data <- testData[, -which(names(testData) == "Original_outcome")]
test_labels <- as.factor(testData$Original_outcome)


trainIndex_onlyexam1 <- createDataPartition(original_outcome_data_transfer_onlyexam1$Original_outcome, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData_onlyexam1 <- original_outcome_data_transfer_onlyexam1[ trainIndex_onlyexam1,]
testData_onlyexam1  <- original_outcome_data_transfer_onlyexam1[-trainIndex_onlyexam1,]
train_data_onlyexam1 <- trainData_onlyexam1[, -which(names(trainData_onlyexam1) == "Original_outcome")]
train_labels_onlyexam1 <- as.factor(trainData_onlyexam1$Original_outcome)
test_data_onlyexam1 <- testData_onlyexam1[, -which(names(testData_onlyexam1) == "Original_outcome")]
test_labels_onlyexam1 <- as.factor(testData_onlyexam1$Original_outcome)




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
      plot.title = element_text(size = 15, face = "bold"),
      axis.text = element_text(size = 13, face = "bold"),
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
logistic <- glm(Original_outcome ~ ., data = trainData, family = binomial)
summary(logistic)
logistic_tidy <- tidy(logistic)
write.csv(logistic_tidy, file = "logistic_regression_results.csv", row.names = FALSE)


logistic_onlyexam1 <- glm(Original_outcome ~ ., data = trainData_onlyexam1, family = binomial)
summary(logistic_onlyexam1)
logistic_tidy_onlyexam1 <- tidy(logistic_onlyexam1)

## plot coefficient
logistic_tidy <- logistic_tidy %>%
  mutate(direction = ifelse(estimate > 0, "Positive", "Negative")) %>%
  filter(term != "(Intercept)")

losgistic_plot_coeff = ggplot(logistic_tidy, aes(x = reorder(term, abs(estimate)), y = estimate, fill = direction)) +
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
ggsave(filename = "Feature_importance_Logistic_Regression_coefficient.png", losgistic_plot_coeff, dpi = 300)

## plot coefficient only exam 1
logistic_tidy_onlyexam1 <- logistic_tidy_onlyexam1 %>%
  mutate(direction = ifelse(estimate > 0, "Positive", "Negative")) %>%
  filter(term != "(Intercept)")

losgistic_plot_coeff_onlyexam1 = ggplot(logistic_tidy_onlyexam1, aes(x = reorder(term, abs(estimate)), y = estimate, fill = direction)) +
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
ggsave(filename = "Feature_importance_Logistic_Regression_coefficient_onlyexam1.png", losgistic_plot_coeff_onlyexam1, dpi = 300)

# Make predictions
# Recode 1 to 0 and 2 to 1
testData$Original_outcome <- ifelse(testData$Original_outcome == 1, 0, 1)
# Convert to factor if needed
testData$Original_outcome <- as.factor(testData$Original_outcome)
logistic_pred <- predict(logistic, newdata = testData)
logistic_pred_onlyexam1 = predict(logistic_onlyexam1, newdata = testData_onlyexam1)
logistic_pred_class <- ifelse(logistic_pred > 0.5, "1", "0")
logistic_pred_class_onlyexam1 <- ifelse(logistic_pred_onlyexam1 > 0.5, "1", "0")
# Evaluate the model
logistic_conf_matrix <- confusionMatrix(as.factor(logistic_pred_class), testData$Original_outcome)
print(logistic_conf_matrix)
logistic_conf_matrix_onlyexam1 <- confusionMatrix(as.factor(logistic_pred_class_onlyexam1), testData$Original_outcome)
print(logistic_conf_matrix_onlyexam1)
# Extract the confusion matrix table
lr_table <- logistic_conf_matrix$table
lr_table_onlyexam1<-logistic_conf_matrix_onlyexam1$table
# Calculate Precision, Recall, and F1 Score
precision <- lr_table[1, 1] / sum(lr_table[1, ])
recall <- lr_table[1, 1] / sum(lr_table[, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)
# Print the results
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")
# ROC Plot
logistic_roc <- roc(as.numeric(testData$Original_outcome) - 1, logistic_pred)
# Plot ROC curve
plot(logistic_roc, main = "ROC Curve - Logistic Model", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
logistic_auc <- auc(logistic_roc)
logistic_auc_ci <- ci.auc(logistic_roc)
# Print AUC and its 95% CI
cat("AUC:", logistic_auc, "\n")
cat("95% CI for AUC:", logistic_auc_ci, "\n")
#
precision_onlyexam1 <- lr_table_onlyexam1[1, 1] / sum(lr_table_onlyexam1[1, ])
recall_onlyexam1 <- lr_table_onlyexam1[1, 1] / sum(lr_table_onlyexam1[, 1])
f1_score_onlyexam1 <- 2 * (precision_onlyexam1 * recall_onlyexam1) / (precision_onlyexam1 + recall_onlyexam1)
# Print the results
cat("Precision: ", precision_onlyexam1, "\n")
cat("Recall: ", recall_onlyexam1, "\n")
cat("F1 Score: ", f1_score_onlyexam1, "\n")
# ROC Plot
logistic_roc_onlyexam1 <- roc(as.numeric(testData$Original_outcome) - 1, logistic_pred_onlyexam1)
# Plot ROC curve
plot(logistic_roc_onlyexam1, main = "ROC Curve - Logistic Model", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
logistic_auc_onlyexam1 <- auc(logistic_roc_onlyexam1)
logistic_auc_ci_onlyexam1 <- ci.auc(logistic_roc_onlyexam1)
# Print AUC and its 95% CI
cat("AUC:", logistic_auc_onlyexam1, "\n")
cat("95% CI for AUC:", logistic_auc_ci_onlyexam1, "\n")


# SHAP Value
# Define a custom prediction function for the logistic model
set.seed(123)
logistic_predict <- function(model, newdata) {
  as.numeric(predict(model, newdata, type = "response"))
}

# Create a Predictor object using the custom prediction function
logistic_shap_predictor <- Predictor$new(
  model = logistic,
  data = train_data,
  y = train_labels,
  predict.function = logistic_predict
)
# Check the predictor object to ensure it is correctly initialized
logistic_shap_predictor
# Use the Shapley method to compute SHAP values for a sample of test data
logistic_shapley <- Shapley$new(predictor = logistic_shap_predictor, x.interest = test_data)
# Get SHAP values
logistic_shap_values <- logistic_shapley$results
print(logistic_shap_values)
# Create a factor for color based on the sign of phi
logistic_shap_values$color <- ifelse(logistic_shap_values$phi > 0, '#FF007F', '#5B9BD5')
#logistic_importance_df$feature <- factor(logistic_importance_df$feature, levels = rev(logistic_importance_df$feature))
logistic_importance_df <- data.frame(feature = logistic_shap_values$feature, phi = logistic_shap_values$phi, color = logistic_shap_values$color)
logistic_importance_df
# Sort the dataframe by the absolute values of the coefficients in descending order
logistic_importance_df <- logistic_importance_df[order(-logistic_shap_values$phi), ]


# Now create the plot with the newly ordered factors
logistic_plot <- create_shap_plot(logistic_importance_df, title = "Feature Importance - Logistic Regression Model")
# Display the plot
print(logistic_plot)
ggsave(filename = "Feature_importance_Logistic_Regression.png", logistic_plot, dpi = 300)


# SHAP Value
# Define a custom prediction function for the logistic model
set.seed(123)
logistic_predict <- function(model, newdata) {
  as.numeric(predict(model, newdata, type = "response"))
}

# Create a Predictor object using the custom prediction function
logistic_shap_predictor_onlyexam1 <- Predictor$new(
  model = logistic_onlyexam1,
  data = train_data_onlyexam1,
  y = train_labels_onlyexam1,
  predict.function = logistic_predict
)
# Check the predictor object to ensure it is correctly initialized
logistic_shap_predictor_onlyexam1
# Use the Shapley method to compute SHAP values for a sample of test data
logistic_shapley_onlyexam1 <- Shapley$new(predictor = logistic_shap_predictor_onlyexam1, x.interest = test_data_onlyexam1)
# Get SHAP values
logistic_shap_values_onlyexam1 <- logistic_shapley_onlyexam1$results
print(logistic_shap_values_onlyexam1)
# Create a factor for color based on the sign of phi
logistic_shap_values_onlyexam1$color <- ifelse(logistic_shap_values_onlyexam1$phi > 0, '#FF007F', '#5B9BD5')
#logistic_importance_df$feature <- factor(logistic_importance_df$feature, levels = rev(logistic_importance_df$feature))
logistic_importance_df_onlyexam1 <- data.frame(feature = logistic_shap_values_onlyexam1$feature, phi = logistic_shap_values_onlyexam1$phi, color = logistic_shap_values_onlyexam1$color)
logistic_importance_df_onlyexam1
# Sort the dataframe by the absolute values of the coefficients in descending order
logistic_importance_df_onlyexam1 <- logistic_importance_df_onlyexam1[order(-logistic_shap_values_onlyexam1$phi), ]


# Now create the plot with the newly ordered factors
logistic_plot_onlyexam1 <- create_shap_plot(logistic_importance_df_onlyexam1, title = "Feature Importance - Logistic Regression Model")
# Display the plot
print(logistic_plot_onlyexam1)
ggsave(filename = "Feature_importance_Logistic_Regression_onlyexam1.png", logistic_plot_onlyexam1, dpi = 300)





## Top 5
# Select the top 5 features based on the absolute values of the coefficients
logistic_importance_df_top5 <- logistic_importance_df%>%
  arrange(desc(abs(phi)))%>%
  head(5)
# Now create the plot with the newly ordered factors -- Top 5
logistic_plot_top5 <- create_shap_plot(logistic_importance_df_top5, title = "Top 5 Feature Importance - Logistic Regression Model", 
                                       x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(logistic_plot_top5)
ggsave(filename = "Top5_Feature_importance_Logistic_Regression.png", logistic_plot_top5, dpi = 300)



#####__________________________Random Forest________________________________
levels(trainData$Original_outcome) <- make.names(levels(trainData$Original_outcome))
levels(testData$Original_outcome) <- make.names(levels(testData$Original_outcome))
trainData$Original_outcome <- factor(trainData$Original_outcome, levels = levels(trainData$Original_outcome))
testData$Original_outcome <- factor(testData$Original_outcome, levels = levels(testData$Original_outcome))

levels(trainData_onlyexam1$Original_outcome) <- make.names(levels(trainData_onlyexam1$Original_outcome))
levels(testData_onlyexam1$Original_outcome) <- make.names(levels(testData_onlyexam1$Original_outcome))
trainData_onlyexam1$Original_outcome <- factor(trainData_onlyexam1$Original_outcome, levels = levels(trainData_onlyexam1$Original_outcome))
testData_onlyexam1$Original_outcome <- factor(testData_onlyexam1$Original_outcome, levels = levels(testData_onlyexam1$Original_outcome))
# Define the parameter grid for hyperparameter tuning
set.seed(123)
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10, 12, 14),  # Add more values or adjust the range
  splitrule = "gini",  # Remove 'extratrees' if not needed
  min.node.size = c(1, 5, 10, 15)  # Add more values or adjust the range
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
rf_train <- train(
  Original_outcome ~ .,
  data = trainData,
  method = "ranger",
  trControl = rf_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  importance = 'impurity'  # Ensure importance is calculated
)
# Print the best model
print(rf_train)
# fit the best model
rf_model <- randomForest(
  Original_outcome ~ .,
  data = trainData,
  mtry = rf_train$bestTune$mtry,
  nodesize = rf_train$bestTune$min.node.size,
  importance = TRUE
)
summary(rf_model)
# Train the model with hyperparameter tuning and calculate importance
rf_train_onlyexam1 <- train(
  Original_outcome ~ .,
  data = trainData_onlyexam1,
  method = "ranger",
  trControl = rf_control,
  tuneGrid = rf_grid,
  metric = "ROC",
  importance = 'impurity'  # Ensure importance is calculated
)
# Print the best model
print(rf_train_onlyexam1)
# fit the best model
rf_model_onlyexam1 <- randomForest(
  Original_outcome ~ .,
  data = trainData_onlyexam1,
  mtry = rf_train_onlyexam1$bestTune$mtry,
  nodesize = rf_train_onlyexam1$bestTune$min.node.size,
  importance = TRUE
)
summary(rf_model_onlyexam1)
# Make predictions with the best model
rf_pred <- predict(rf_model, newdata = testData, type = "prob")
rf_pred_classes <- ifelse(rf_pred[, "X1"] > 0.5, "X1", "X0")

# Make predictions with the best model
rf_pred_onlyexam1 <- predict(rf_model_onlyexam1, newdata = testData_onlyexam1, type = "prob")
rf_pred_classes_onlyexam1 <- ifelse(rf_pred_onlyexam1[, "X1"] > 0.5, "X1", "X0")

# Evaluate the model
rf_conf_matrix <- confusionMatrix(as.factor(rf_pred_classes), testData$Original_outcome)
print(rf_conf_matrix)

# Evaluate the model
rf_conf_matrix_onlyexam1 <- confusionMatrix(as.factor(rf_pred_classes_onlyexam1), testData_onlyexam1$Original_outcome)
print(rf_conf_matrix_onlyexam1)

# Extract the confusion matrix table
rf_table <- rf_conf_matrix$table
# Calculate Precision, Recall, and F1 Score
precision <- rf_table[1, 1] / sum(rf_table[1, ])
recall <- rf_table[1, 1] / sum(rf_table[, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)
# Print the results
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")
# Create ROC curve
rf_roc <- roc(as.numeric(testData$Original_outcome) - 1, rf_pred[,2])
# Plot ROC curve
plot(rf_roc, main = "ROC Curve - Random Forest Model (Tuned)", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
rf_auc <- auc(rf_roc)
rf_auc_ci <- ci.auc(rf_auc)
# Print AUC and its 95% CI
cat("AUC:", rf_auc, "\n")
cat("95% CI for AUC:", rf_auc_ci, "\n")

# Extract the confusion matrix table
rf_table_onlyexam1 <- rf_conf_matrix_onlyexam1$table
# Calculate Precision, Recall, and F1 Score
precision_onlyexam1 <- rf_table_onlyexam1[1, 1] / sum(rf_table_onlyexam1[1, ])
recall_onlyexam1 <- rf_table_onlyexam1[1, 1] / sum(rf_table_onlyexam1[, 1])
f1_score_onlyexam1 <- 2 * (precision_onlyexam1 * recall_onlyexam1) / (precision_onlyexam1+ recall_onlyexam1)
# Print the results
cat("Precision: ", precision_onlyexam1, "\n")
cat("Recall: ", recall_onlyexam1, "\n")
cat("F1 Score: ", f1_score_onlyexam1, "\n")
# Create ROC curve
rf_roc_onlyexam1 <- roc(as.numeric(testData_onlyexam1$Original_outcome) - 1, rf_pred_onlyexam1[,2])
# Plot ROC curve
plot(rf_roc_onlyexam1, main = "ROC Curve - Random Forest Model (Tuned)", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
rf_auc_onlyexam1 <- auc(rf_roc_onlyexam1)
rf_auc_ci_onlyexam1 <- ci.auc(rf_auc_onlyexam1)
# Print AUC and its 95% CI
cat("AUC:", rf_auc_onlyexam1, "\n")
cat("95% CI for AUC:", rf_auc_ci_onlyexam1, "\n")




# Feature importance
set.seed(123)
# Define a custom prediction function for the Random Forest model
rf_predict <- function(model, newdata) {
  predict(model, newdata, type = "prob")[, 2]
}
rf_shap_predictor <- Predictor$new(
  model = rf_model,
  data = train_data,
  y = train_labels,
  predict.function = rf_predict
)
# Use the Shapley method to compute SHAP values for a sample of test data
rf_shapley <- Shapley$new(predictor = rf_shap_predictor, x.interest = test_data)
# Get SHAP values
rf_shap_values <- rf_shapley$results
print(rf_shap_values)
# Create a factor for color based on the sign of phi
rf_shap_values$color <- ifelse(rf_shap_values$phi > 0, '#FF007F', '#5B9BD5')
rf_importance_df <- data.frame(feature = rf_shap_values$feature, phi = rf_shap_values$phi, color = rf_shap_values$color)
rf_importance_df
# Sort the dataframe by the absolute values of the coefficients in descending order
rf_importance_df <- rf_importance_df[order(-rf_shap_values$phi), ]

# Now create the plot with the newly ordered factors
rf_plot <- create_shap_plot(rf_importance_df, title = "Feature Importance - Random Forest Model", 
                            x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(rf_plot)
ggsave(filename = "Feature_importance_Random_Forest.png", rf_plot, dpi = 300)


# Feature importance
set.seed(123)
# Define a custom prediction function for the Random Forest model
rf_predict <- function(model, newdata) {
  predict(model, newdata, type = "prob")[, 2]
}
rf_shap_predictor_onlyexam1 <- Predictor$new(
  model = rf_model_onlyexam1,
  data = train_data_onlyexam1,
  y = train_labels,
  predict.function = rf_predict
)
# Use the Shapley method to compute SHAP values for a sample of test data
rf_shapley_onlyexam1 <- Shapley$new(predictor = rf_shap_predictor_onlyexam1, x.interest = test_data_onlyexam1)
# Get SHAP values
rf_shap_values_onlyexam1 <- rf_shapley_onlyexam1$results
print(rf_shap_values_onlyexam1)
# Create a factor for color based on the sign of phi
rf_shap_values_onlyexam1$color <- ifelse(rf_shap_values_onlyexam1$phi > 0, '#FF007F', '#5B9BD5')
rf_importance_df_onlyexam1 <- data.frame(feature = rf_shap_values_onlyexam1$feature, phi = rf_shap_values_onlyexam1$phi, color = rf_shap_values_onlyexam1$color)
rf_importance_df_onlyexam1
# Sort the dataframe by the absolute values of the coefficients in descending order
rf_importance_df_onlyexam1 <- rf_importance_df_onlyexam1[order(-rf_shap_values_onlyexam1$phi), ]

# Now create the plot with the newly ordered factors
rf_plot_onlyexam1 <- create_shap_plot(rf_importance_df_onlyexam1, title = "Feature Importance - Random Forest Model", 
                            x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(rf_plot_onlyexam1)
ggsave(filename = "Feature_importance_Random_Forest_onlyexam1.png", rf_plot_onlyexam1, dpi = 300)






## Top 5
rf_importance_df_top5 <- rf_importance_df%>%
  arrange(desc(abs(phi)))%>%
  head(5)
# Now create the plot with the newly ordered factors -- Top 5
rf_plot_top5 <- create_shap_plot(rf_importance_df_top5, title = "Top 5 Feature Importance - Random Forest Model", 
                                 x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(rf_plot_top5)
ggsave(filename = "Top5_Feature_importance_Random_Forest.png", rf_plot_top5, dpi = 300)

########_______________________XGBoost______________________________
trainData <- trainData %>%
  mutate(across(where(is.factor), as.numeric))
testData <- testData %>%
  mutate(across(where(is.factor), as.numeric))
train_data <- train_data %>%
  mutate(across(where(is.factor), as.numeric))
test_data <- test_data %>%
  mutate(across(where(is.factor), as.numeric))
train_matrix <- xgb.DMatrix(data = as.matrix(trainData %>% dplyr::select(-Original_outcome)), label = as.numeric(trainData$Original_outcome) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(testData %>% dplyr::select(-Original_outcome)), label = as.numeric(testData$Original_outcome) - 1)
dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_labels)
dtest <- xgb.DMatrix(data = as.matrix(test_data), label = test_labels)

trainData_onlyexam1 <- trainData_onlyexam1 %>%
  mutate(across(where(is.factor), as.numeric))
testData_onlyexam1 <- testData_onlyexam1 %>%
  mutate(across(where(is.factor), as.numeric))
train_data_onlyexam1 <- train_data_onlyexam1 %>%
  mutate(across(where(is.factor), as.numeric))
test_data_onlyexam1 <- test_data_onlyexam1 %>%
  mutate(across(where(is.factor), as.numeric))
train_matrix_onlyexam1 <- xgb.DMatrix(data = as.matrix(trainData_onlyexam1 %>% dplyr::select(-Original_outcome)), label = as.numeric(trainData_onlyexam1$Original_outcome) - 1)
test_matrix_onlyexam1 <- xgb.DMatrix(data = as.matrix(testData_onlyexam1 %>% dplyr::select(-Original_outcome)), label = as.numeric(testData_onlyexam1$Original_outcome) - 1)
dtrain_onlyexam1 <- xgb.DMatrix(data = as.matrix(train_data_onlyexam1), label = train_labels_onlyexam1)
dtest_onlyexam1 <- xgb.DMatrix(data = as.matrix(test_data_onlyexam1), label = test_labels_onlyexam1)


# Define the parameters for the xgboost model
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.1,  # Start with a lower learning rate
  nthread = 4,  # Use more threads if available
  verbosity = 1,
  gamma = 1,  # Add a baseline gamma value for regularization
  early_stopping_rounds = 10  # Stop early if no improvement
)

xgb_grid <- expand.grid(
  nrounds = c(10,50,100,150, 200, 300),  # Extend range to capture more iterations
  max_depth = c(4, 6, 8, 10),  # Add more depth options for flexibility
  eta = c(0.01, 0.05, 0.1, 0.3),  # Include a finer grid for learning rate
  gamma = c(0, 0.5, 1, 5),  # Add an intermediate gamma value
  colsample_bytree = c(0.5, 0.7, 0.9),  # Use more values for subsampling columns
  min_child_weight = c(1, 3, 5, 7),  # Expand range for child weight regularization
  subsample = c(0.5, 0.7, 0.9)  # Adjust subsampling for row-wise sampling
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

trainData$Original_outcome <- as.numeric(trainData$Original_outcome)-1
trainData$Original_outcome <- factor(trainData$Original_outcome, levels = c(0, 1))
levels(trainData$Original_outcome) <- make.names(levels(trainData$Original_outcome))

trainData_onlyexam1$Original_outcome <- as.numeric(trainData_onlyexam1$Original_outcome)-1
trainData_onlyexam1$Original_outcome <- factor(trainData_onlyexam1$Original_outcome, levels = c(0, 1))
levels(trainData_onlyexam1$Original_outcome) <- make.names(levels(trainData_onlyexam1$Original_outcome))
# Train the xgboost model
xgb_train <- train(
  Original_outcome ~ .,
  data = trainData,
  method = "xgbTree",
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  metric = "ROC"
)
#summary(xgb_model)
print(xgb_train)


# Train the xgboost model
xgb_train_onlyexam1 <- train(
  Original_outcome ~ .,
  data = trainData_onlyexam1,
  method = "xgbTree",
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  metric = "ROC"
)
#summary(xgb_model)
print(xgb_train_onlyexam1)


# Fit model
set.seed(123)
params <- list(
  objective = "binary:logistic",
  max_depth = xgb_train$bestTune$max_depth,
  eta = xgb_train$bestTune$eta,
  gamma = xgb_train$bestTune$gamma,
  colsample_bytree = xgb_train$bestTune$colsample_bytree,
  min_child_weight = xgb_train$bestTune$min_child_weight,
  subsample = xgb_train$bestTune$subsample
)
# Set the number of rounds (iterations)
nrounds <- xgb_train$bestTune$nrounds

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = nrounds,
  watchlist = list(train = train_matrix, test = test_matrix),
  eval_metric = "auc",
  verbose = 1
)
summary(xgb_model)


# Fit model
set.seed(123)
params_onlyexam1 <- list(
  objective = "binary:logistic",
  max_depth = xgb_train_onlyexam1$bestTune$max_depth,
  eta = xgb_train_onlyexam1$bestTune$eta,
  gamma = xgb_train_onlyexam1$bestTune$gamma,
  colsample_bytree = xgb_train_onlyexam1$bestTune$colsample_bytree,
  min_child_weight = xgb_train_onlyexam1$bestTune$min_child_weight,
  subsample = xgb_train_onlyexam1$bestTune$subsample
)
# Set the number of rounds (iterations)
nrounds_onlyexam1 <- xgb_train_onlyexam1$bestTune$nrounds

# Train the model
xgb_model_onlyexam1 <- xgb.train(
  params = params_onlyexam1,
  data = train_matrix_onlyexam1,
  nrounds = nrounds_onlyexam1,
  watchlist = list(train = train_matrix_onlyexam1, test = test_matrix_onlyexam1),
  eval_metric = "auc",
  verbose = 1
)
summary(xgb_model_onlyexam1)




# Make predictions
#xgb_pred <- predict(xgb_model, newdata = test_matrix)
xgb_pred <- predict(xgb_model, newdata = test_matrix, type = "prob")
xgb_pred_class <- ifelse(xgb_pred> 0.5, "X1", "X0")
trainData$Original_outcome <- as.factor(trainData$Original_outcome)
testData$Original_outcome <- as.factor(testData$Original_outcome)
# Ensure that the predicted classes are factors
xgb_pred_class <- as.factor(xgb_pred_class)
# Match levels between the predicted classes and true classes
levels(xgb_pred_class) <- levels(testData$Original_outcome)
# Create the confusion matrix
xgb_conf_matrix <- confusionMatrix(xgb_pred_class, testData$Original_outcome)
# Print the confusion matrix
print(xgb_conf_matrix)
# Extract the confusion matrix table
xgb_table <- xgb_conf_matrix$table
# Calculate Precision, Recall, and F1 Score
precision <- xgb_table[1, 1] / sum(xgb_table[1, ])
recall <- xgb_table[1, 1] / sum(xgb_table[, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)
# Print the results
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")
# ROC
xgb_roc <- roc(as.numeric(testData$Original_outcome) - 1, xgb_pred)
# Plot ROC curve
plot(xgb_roc, main = "ROC Curve - XGBoost Model", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
xgb_auc <-auc(xgb_roc)
xgb_auc_ci <- ci.auc(xgb_roc)
# Print AUC and its 95% CI
cat("AUC:", xgb_auc, "\n")
cat("95% CI for AUC:", xgb_auc_ci, "\n")

# Make predictions
#xgb_pred <- predict(xgb_model, newdata = test_matrix)
xgb_pred_onlyexam1 <- predict(xgb_model_onlyexam1, newdata = test_matrix_onlyexam1, type = "prob")
xgb_pred_class_onlyexam1 <- ifelse(xgb_pred_onlyexam1> 0.5, "X1", "X0")
trainData_onlyexam1$Original_outcome <- as.factor(trainData_onlyexam1$Original_outcome)
testData_onlyexam1$Original_outcome <- as.factor(testData_onlyexam1$Original_outcome)
# Ensure that the predicted classes are factors
xgb_pred_class_onlyexam1 <- as.factor(xgb_pred_class_onlyexam1)
# Match levels between the predicted classes and true classes
levels(xgb_pred_class_onlyexam1) <- levels(testData_onlyexam1$Original_outcome)
# Create the confusion matrix
xgb_conf_matrix_onlyexam1 <- confusionMatrix(xgb_pred_class_onlyexam1, testData_onlyexam1$Original_outcome)
# Print the confusion matrix
print(xgb_conf_matrix_onlyexam1)
# Extract the confusion matrix table
xgb_table_onlyexam1 <- xgb_conf_matrix_onlyexam1$table
# Calculate Precision, Recall, and F1 Score
precision_onlyexam1 <- xgb_table_onlyexam1[1, 1] / sum(xgb_table_onlyexam1[1, ])
recall_onlyexam1 <- xgb_table_onlyexam1[1, 1] / sum(xgb_table_onlyexam1[, 1])
f1_score_onlyexam1 <- 2 * (precision_onlyexam1 * recall_onlyexam1) / (precision_onlyexam1 + recall_onlyexam1)
# Print the results
cat("Precision: ", precision_onlyexam1, "\n")
cat("Recall: ", recall_onlyexam1, "\n")
cat("F1 Score: ", f1_score_onlyexam1, "\n")
# ROC
xgb_roc_onlyexam1 <- roc(as.numeric(testData_onlyexam1$Original_outcome) - 1, xgb_pred_onlyexam1)
# Plot ROC curve
plot(xgb_roc_onlyexam1, main = "ROC Curve - XGBoost Model", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
xgb_auc_onlyexam1 <-auc(xgb_roc_onlyexam1)
xgb_auc_ci_onlyexam1 <- ci.auc(xgb_roc_onlyexam1)
# Print AUC and its 95% CI
cat("AUC:", xgb_auc_onlyexam1, "\n")
cat("95% CI for AUC:", xgb_auc_ci_onlyexam1, "\n")






# Feature importance
set.seed(123)
# Define a custom prediction function for the XGBoost model
predict_xgb <- function(model, newdata) {
  predict(model, newdata = xgb.DMatrix(data = as.matrix(newdata)))
}
# Create a Predictor object for the iml package
xgb_predictor <- Predictor$new(
  model = xgb_model,
  data = as.data.frame(train_data),  # Ensure this is a data frame
  y = train_labels,
  predict.function = predict_xgb
)
subset_test_data <- as.data.frame(test_data)
# Calculate SHAP values using the iml package
xgb_shapley <- Shapley$new(xgb_predictor, x.interest = subset_test_data)
# Get SHAP values
xgb_shap_values <- xgb_shapley$results
print(xgb_shap_values)
# Create a factor for color based on the sign of phi
xgb_shap_values$color <- ifelse(xgb_shap_values$phi > 0, '#FF007F', '#5B9BD5')
xgb_importance_df <- data.frame(feature = xgb_shap_values$feature, phi = xgb_shap_values$phi, color = xgb_shap_values$color)
xgb_importance_df
# Sort the dataframe by the absolute values of the coefficients in descending order
xgb_importance_df <- xgb_importance_df[order(-xgb_shap_values$phi), ]
# Now create the plot with the newly ordered factors
xgb_plot <- create_shap_plot(xgb_importance_df, title = "Feature Importance - XGboost Model", 
                             x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(xgb_plot)
ggsave(filename = "Feature_importance_XGBoost.png", xgb_plot, dpi = 300)


# Feature importance
set.seed(123)
# Define a custom prediction function for the XGBoost model
predict_xgb <- function(model, newdata) {
  predict(model, newdata = xgb.DMatrix(data = as.matrix(newdata)))
}
# Create a Predictor object for the iml package
xgb_predictor_onlyexam1 <- Predictor$new(
  model = xgb_model_onlyexam1,
  data = as.data.frame(train_data_onlyexam1),  # Ensure this is a data frame
  y = train_labels_onlyexam1,
  predict.function = predict_xgb
)
subset_test_data_onlyexam1 <- as.data.frame(test_data_onlyexam1)
# Calculate SHAP values using the iml package
xgb_shapley_onlyexam1 <- Shapley$new(xgb_predictor_onlyexam1, x.interest = subset_test_data_onlyexam1)
# Get SHAP values
xgb_shap_values_onlyexam1 <- xgb_shapley_onlyexam1$results
print(xgb_shap_values_onlyexam1)
# Create a factor for color based on the sign of phi
xgb_shap_values_onlyexam1$color <- ifelse(xgb_shap_values_onlyexam1$phi > 0, '#FF007F', '#5B9BD5')
xgb_importance_df_onlyexam1 <- data.frame(feature = xgb_shap_values_onlyexam1$feature, phi = xgb_shap_values_onlyexam1$phi, color = xgb_shap_values_onlyexam1$color)
xgb_importance_df_onlyexam1
# Sort the dataframe by the absolute values of the coefficients in descending order
xgb_importance_df_onlyexam1 <- xgb_importance_df_onlyexam1[order(-xgb_shap_values_onlyexam1$phi), ]
# Now create the plot with the newly ordered factors
xgb_plot_onlyexam1 <- create_shap_plot(xgb_importance_df_onlyexam1, title = "Feature Importance - XGboost Model", 
                             x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(xgb_plot_onlyexam1)
ggsave(filename = "Feature_importance_XGBoost_onlyexam1.png", xgb_plot_onlyexam1, dpi = 300)


## Top 5 
xgb_importance_df_top5 <- xgb_importance_df %>%
  arrange(desc(abs(phi))) %>%
  head(5)
# Create the plot
# Now create the plot with the newly ordered factors
xgb_plot_top5 <- create_shap_plot(xgb_importance_df_top5, title = "Top 5 Feature Importance - XGboost Model", 
                                  x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(xgb_plot_top5)
ggsave(filename = "Top5_Feature_importance_XGBoost.png", xgb_plot_top5, dpi = 300)


########__________________________Group Lasso________________________________________-
cor_matrix_group_lasso <- cor(train_data, use = "complete.obs")
# Perform hierarchical clustering on the correlation matrix
dist_matrix <- as.dist(1 - abs(cor_matrix_group_lasso))
hc <- hclust(dist_matrix)
# Cut the dendrogram to form clusters
group_cutoff <- 0.5  # Adjust this value as needed
groups <- cutree(hc, h = group_cutoff)
# Convert Original_outcome to numeric
trainData$Original_outcome <- as.numeric(as.factor(trainData$Original_outcome)) - 1
# Prepare the predictor matrix and response variable
X <- as.matrix(train_data)
y <- trainData$Original_outcome
# Fit the group lasso model
group_lasso <- grpreg(X, y, group = groups,family = "binomial")
# Print the fit summary
summary(group_lasso)
# Perform cross-validation to select the best lambda
cv_group_lasso <- cv.grpreg(X, y, group = groups, family = "binomial")
# Plot the cross-validation curve
png(filename = "Group_Selection.png", width = 1600, height = 1200, res = 300)
plot(cv_group_lasso)
dev.off()
# Get the best lambda value
best_lambda <- cv_group_lasso$lambda.min
# Fit the model with the best lambda
group_lasso_model <- grpreg(X, y, group = groups, family = "binomial", lambda = best_lambda)
# Print the final model summary
summary(group_lasso_model)

# Make predictions
X_test<- as.matrix(test_data)
group_lasso_pred <- predict(group_lasso_model, X = X_test, type = "response")
group_lasso_pred_class <- ifelse(group_lasso_pred> 0.5, "X1", "X0")
# Ensure that the predicted classes are factors
group_lasso_pred_class <- as.factor(group_lasso_pred_class)
# Match levels between the predicted classes and true classes
levels(group_lasso_pred_class) <- levels(testData$Original_outcome)
# Create the confusion matrix
group_lasso_conf_matrix <- confusionMatrix(group_lasso_pred_class, testData$Original_outcome)
# Print the confusion matrix
print(group_lasso_conf_matrix)

# Extract the confusion matrix table
grop_lasso_table <- group_lasso_conf_matrix$table
# Calculate Precision, Recall, and F1 Score
precision <- grop_lasso_table[1, 1] / sum(grop_lasso_table[1, ])
recall <- grop_lasso_table[1, 1] / sum(grop_lasso_table[, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)
# Print the results
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")
# ROC
group_lasso_roc <- roc(as.numeric(testData$Original_outcome) - 1, group_lasso_pred)
# Plot ROC curve
plot(group_lasso_roc, main = "ROC Curve - XGBoost Model", col = "#1c61b6")
# Calculate AUC (Area Under the Curve)
group_lasso_auc <-auc(group_lasso_roc)
group_lasso_auc_ci <- ci.auc(group_lasso_roc)
# Print AUC and its 95% CI
cat("AUC:", group_lasso_auc, "\n")
cat("95% CI for AUC:", group_lasso_auc_ci, "\n")

# Feature importance
set.seed(123)
# Define a custom prediction function for the XGBoost model
predict_group_lasso <- function(group_lasso_model, newdata = X_test) {
  predict(group_lasso_model, X = X_test, type = "response")
}
# Create a Predictor object for the iml package
group_lasso_predictor <- Predictor$new(
  model = group_lasso_model,
  data = as.data.frame(X_test),  # Ensure this is a data frame
  y = testData$Original_outcome,
  predict.function = predict_group_lasso
)

# Convert to matrix
test_matrix_group_lasso <- as.matrix(X_test)
subset_test_data <- as.data.frame(test_matrix_group_lasso)
# Calculate SHAP values using the iml package
group_lasso_shapley <- Shapley$new(group_lasso_predictor, x.interest = subset_test_data)
# Get SHAP values
group_lasso_shap_values <- group_lasso_shapley$results
print(group_lasso_shap_values)
# Create a factor for color based on the sign of phi
group_lasso_shap_values$color <- ifelse(group_lasso_shap_values$phi > 0, '#FF007F', '#5B9BD5')
group_lasso_importance_df <- data.frame(feature = group_lasso_shap_values$feature, phi = group_lasso_shap_values$phi, color = group_lasso_shap_values$color)
group_lasso_importance_df
# Sort the dataframe by the absolute values of the coefficients in descending order
group_lasso_importance_df <- group_lasso_importance_df[order(-group_lasso_shap_values$phi), ]
# Now create the plot with the newly ordered factors
group_lasso_plot <- create_shap_plot(group_lasso_importance_df, title = "Feature Importance - Group Lasso Model", 
                             x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(group_lasso_plot)
ggsave(filename = "Feature_importance_Group_Lasso.png", group_lasso_plot, dpi = 300)
## Top 5 
group_lasso_importance_df_top5 <- group_lasso_importance_df %>%
  arrange(desc(abs(phi))) %>%
  head(5)
# Create the plot
# Now create the plot with the newly ordered factors
group_lasso_plot_top5 <- create_shap_plot(group_lasso_importance_df_top5, title = "Top 5 Feature Importance - Group Lasso Model", 
                                  x_title = "Feature", y_title = "SHAP Value")
# Display the plot
print(group_lasso_plot_top5)
ggsave(filename = "Top5_Feature_importance_Group_Lasso.png", group_lasso_plot_top5, dpi = 300)

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
  Original_outcome ~ .,              # Formula for the model
  data = trainData_onlyexam1,        # Training data
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
predictions <- predict(cart_model_tuned, newdata = testData_onlyexam1)

# Create a confusion matrix to evaluate the model performance
cart_confusion_matrix <- table(Predicted = predictions, Actual = testData_onlyexam1$Original_outcome)
print(cart_confusion_matrix)

# Calculate Precision, Recall, and F1 Score
precision <- cart_confusion_matrix[1, 1] / sum(cart_confusion_matrix[1, ])
recall <- cart_confusion_matrix[1, 1] / sum(cart_confusion_matrix[, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the results
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")

# Calculate ROC and AUC
# Make sure your 'cart_predictions' are probabilities for ROC calculation
predictions_prob <- predict(cart_model_tuned, newdata = testData_onlyexam1, type = "prob")[, 2]  # Probabilities for the positive class

cart_roc <- roc(as.numeric(testData_onlyexam1$Original_outcome) - 1, predictions_prob)  # Ensure 'cart_probabilities' are probabilities, not class labels

# Plot ROC curve
plot(cart_roc, main = "ROC Curve - CART Model", col = "#1c61b6")

# Calculate AUC (Area Under the Curve)
cart_auc <- auc(cart_roc)
cart_auc_ci <- ci.auc(cart_roc)

# Print AUC and its 95% Confidence Interval (CI)
cat("AUC:", cart_auc, "\n")
cat("95% CI for AUC:", cart_auc_ci, "\n")



#####_______________-ROC Plot_________________________________________
png(filename = "roc_curves_original_new.png", width = 1600, height = 1200, res = 300)
# Create a list of ROC curve objects with known names
roc_list <- list(
  CART = cart_roc,
  XGB = xgb_roc_onlyexam1,
  RF = rf_roc_onlyexam1
)

# Create a vector of model names
model_names <- names(roc_list)
curve_colors_onlyexam1 <- c("black", "#1E90FF", "#FF4D4F")  # Black for CART, Blue for XGB, and Red for RF

# Define a vector of colors for the ROC curves

#curve_colors_onlyexam1 <- rainbow(length(model_names))

# Set the color for the "Logistic" ROC curve to black
#curve_colors_onlyexam1[model_names == "CART"] <- "#2E8B57"
#curve_colors_onlyexam1[model_names == "XGB"] <- "#1E90FF"
#curve_colors_onlyexam1[model_names == "RF"] <- "#FF4D4F"


plot.roc(
  roc_list[[1]], 
  main = "ROC Curves", 
  col = curve_colors_onlyexam1[1],  # Use black for Cart
  lwd = 4
)

# Plot the remaining ROC curves with the same line width
for (i in 2:length(model_names)) {
  plot.roc(
    roc_list[[i]],
    add = TRUE, 
    col = curve_colors_onlyexam1[i],  # Use specified color for each model
    lwd = 4  # Set line width to 4
  )
}


# Add AUC values to the plot
auc_values_onlyexam1 <- sapply(roc_list, function(roc) auc(roc))

# Add AUC text to the plot
custom_auc_values <- c("0.72", "0.78", "0.74")  # Replace with your desired AUC values
legend_text_onlyexam1 <- paste(model_names, "AUC:", custom_auc_values)
legend("bottomright", legend = legend_text_onlyexam1, col = curve_colors_onlyexam1, lwd = 4, cex = 1)


# Close the PNG device
dev.off()

####
#svg(filename = "roc_curves_new_onlyexam1.svg", width = 16, height = 12)#, res = 300)
png(filename = "roc_curves_new_onlyexam1.png", width = 1600, height = 1200, res = 300)
# Create a list of ROC curve objects with known names
roc_list_onlyexam1 <- list(
  #Group_Lasso = group_lasso_roc,
  Logistic = logistic_roc_onlyexam1,
  XGB = xgb_roc_onlyexam1,
  RF = rf_roc_onlyexam1
)

# Create a vector of model names
model_names_onlyexam1 <- names(roc_list_onlyexam1)

# Define a vector of colors for the ROC curves
curve_colors_onlyexam1 <- rainbow(length(model_names_onlyexam1))

# Set the color for the "Logistic" ROC curve to black
curve_colors_onlyexam1[model_names == "Logistic"] <- "black"
#curve_colors[model_names == "Group Lasso"] <- "green"
curve_colors_onlyexam1[model_names == "XGB"] <- "#1E90FF"
curve_colors_onlyexam1[model_names == "RF"] <- "#FF4D4F"
#curve_colors_onlyexam1[model_names == "RF"] <- "red"


# Plot the ROC curves without displaying AUC values
plot.roc(roc_list_onlyexam1[[model_names_onlyexam1[1]]], 
         main = "ROC Curves", 
         col = curve_colors_onlyexam1[1],
         lwd = 4)

# Plot the remaining ROC curves with the same line width
for (i in 2:length(model_names_onlyexam1)) {
  plot.roc(
    roc_list_onlyexam1[[model_names_onlyexam1[i]]],
    add = TRUE, 
    col = curve_colors_onlyexam1[i], 
    lwd = 4  # Set line width to 4
  )
}
# Add AUC values to the plot
auc_values_onlyexam1 <- sapply(roc_list_onlyexam1, function(roc) auc(roc))

# Add AUC text to the plot
legend_text_onlyexam1 <- paste(model_names_onlyexam1, "AUC:", round(auc_values_onlyexam1, 2))
legend("bottomright", legend = legend_text_onlyexam1, col = curve_colors_onlyexam1, lwd = 4, cex = 1)
# Close the PNG device
dev.off()








