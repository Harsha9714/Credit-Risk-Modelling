# Load necessary libraries
library(caret)
library(ROCR)
library(lme4)
# Read your training and validation data
train_data <- read.csv("benchmark_training_loan_data.csv")
valid_data <- read.csv("benchmark_validation_loan_data.csv")

# Build a GLM model using training data
glm_model <- glm(repay_fail ~ loan_amnt + term + int_rate + emp_length + home_ownership + annual_inc + verification_status + purpose + dti + delinq_2yrs + inq_last_6mths + open_acc + pub_rec + revol_bal + revol_util + total_acc + credit_age_yrs, data = train_data, family = binomial)

# Predict on training set for new model
train_pred_prob <- predict(glm_model, newdata = train_data, type = "response")

# Predict on validation set for new model
valid_pred_prob <- predict(glm_model, newdata = valid_data, type = "response")

# Create prediction objects
pred_obj_train <- prediction(train_pred_prob, train_data$repay_fail)
pred_obj_valid <- prediction(valid_pred_prob, valid_data$repay_fail)

# Create performance objects
perf_obj_train <- performance(pred_obj_train, "tpr", "fpr")
perf_obj_valid <- performance(pred_obj_valid, "tpr", "fpr")


# Set up the plotting window to have 1 row and 2 columns
par(mfrow=c(1, 2))

# Plot ROC curve for training set
plot(perf_obj_train, main="ROC Curve for Training Set", col=2, lwd=2)
abline(a=0, b=1, lty=2, col="gray")

# Calculate Gini score for training set
auc_train <- as.numeric(performance(pred_obj_train, measure = "auc")@y.values)
gini_train = 2 * auc_train - 1
print(paste("Gini Score for training set: ", gini_train))

# Plot ROC curve for validation set
plot(perf_obj_valid, main="ROC Curve for Validation Set", col=3, lwd=2)
abline(a=0, b=1, lty=2, col="gray")

# Calculate Gini score for validation set
auc_valid <- as.numeric(performance(pred_obj_valid, measure = "auc")@y.values)
gini_valid = 2 * auc_valid - 1
print(paste("Gini Score for validation set: ", gini_valid))


# Model Summary to check important variables
glm_summary <- summary(glm_model)

# Extract p-values for each variable
p_values <- glm_summary$coefficients[, 4]

# Identify important variables based on p-value < 0.05
important_vars <- names(which(p_values < 0.05))

# Print important variables
print("Important Variables:")
print(important_vars)


# Extract coefficients for important variables
important_coeff <- glm_summary$coefficients[important_vars, 1]

# Interpret important variables
print("Interpretation of Important Variables:")
for (var in important_vars) {
  coeff <- important_coeff[var]
  if (coeff > 0) {
    print(paste(var, "increases the likelihood of loan default."))
  } else {
    print(paste(var, "decreases the likelihood of loan default."))
  }
}


library(lme4)

extended_version_data <- read.csv("extendend_version_loan_data.csv")

# Convert issue_d and addr_state to factors 
extended_version_data$issue_d <- as.factor(extended_version_data$issue_d)
extended_version_data$addr_state <- as.factor(extended_version_data$addr_state)

# Standardize continuous variables
vars_to_standardize <- c("loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "credit_age_yrs")
extended_version_data[vars_to_standardize] <- scale(extended_version_data[vars_to_standardize])

# Take a subset of the data for quick testing
subset_data <- extended_version_data[1:1000,]

# Run GLMM model on subset
glmm_model_subset <- glmer(repay_fail ~ loan_amnt + int_rate + (1|issue_d) + (1|addr_state), data = subset_data, family = binomial)

# Summary of the model to check important variables
glmm_summary_subset <- summary(glmm_model_subset)
print(glmm_summary_subset)



# Run GLMM model on full data
glmm_model_full <- glmer(repay_fail ~ loan_amnt + int_rate + (1|issue_d) + (1|addr_state), data = extended_version_data, family = binomial)

# Summary of the model to check important variables
glmm_summary_full <- summary(glmm_model_full)
print(glmm_summary_full)
