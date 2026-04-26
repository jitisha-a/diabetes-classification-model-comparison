############################################################
# Diabetes Classification using KNN, Decision Tree, Logistic Regression
# Choosing the best model for the given data set
############################################################

############################################################
# load packages
############################################################

library(class)
library(rpart)
library(rpart.plot)
library(pROC)

############################################################
# read data
############################################################

setwd("~/Downloads/DATA_SCI")
data <- read.csv("diabetes-dataset.csv", header = TRUE)

names(data)
head(data)
dim(data)
str(data)

############################################################
# part I: EDA
############################################################

# ---------------------------
# response variable
# ---------------------------
table(data$diabetes)
prop.table(table(data$diabetes))

# ---------------------------
# categorical variables vs diabetes
# ---------------------------

# hypertension
table_hypertension <- table(data$hypertension, data$diabetes)
table_hypertension
prop_hypertension <- prop.table(table_hypertension, margin = 1)
prop_hypertension
# rate of having diabetes and hypertension - 0.27895792
# rate of having diabetes but no hypertension - 0.06930768

# heart disease
table_heart_disease <- table(data$heart_disease, data$diabetes)
table_heart_disease
prop_heart_disease <- prop.table(table_heart_disease, margin = 1)
prop_heart_disease
# rate of having diabetes and heart disease - 0.32141045
# rate of having diabetes but no heart disease - 0.07529826

# gender
table_gender <- table(data$gender, data$diabetes)
table_gender
prop_gender <- prop.table(table_gender, margin = 1)
prop_gender
# rate of having diabetes and male - 0.09748974
# rate of having diabetes and female - 0.07618869

# smoking history
table_smoking <- table(data$smoking_history, data$diabetes)
table_smoking
prop_smoking <- prop.table(table_smoking, margin = 1)
prop_smoking

# rate of having diabetes and current  - 0.10208917
# rate of having diabetes and ever  - 0.11788212
# rate of having diabetes and former - 0.17001711
# rate of having diabetes and never - 0.09534122
# rate of having diabetes and No Info - 0.04059638
# rate of having diabetes and not current - 0.10702652

# given the similarities in the rates of:
   # current, ever and former
   # never and not current

# it was decided to regroup smoking_history into 3 categories:
# Yes = current, ever, former
# No = never, not current
# No info = No Info 

data$smoking_history <- ifelse(data$smoking_history %in% c("current", "ever", "former"), "Yes",
                               ifelse(data$smoking_history %in% c("never", "not current"), "No",
                                      ifelse(data$smoking_history %in% c("No Info"), "No info",
                                             data$smoking_history)))

data$smoking_history <- factor(data$smoking_history, levels = c("Yes", "No", "No info"))

# check
str(data)

# ---------------------------
# numerical variables vs diabetes
# ---------------------------

boxplot(age ~ diabetes, data = data,
        main = "Age by Diabetes Status",
        xlab = "Diabetes", ylab = "Age")

boxplot(bmi ~ diabetes, data = data,
        main = "BMI by Diabetes Status",
        xlab = "Diabetes", ylab = "BMI")

boxplot(HbA1c_level ~ diabetes, data = data,
        main = "HbA1c Level by Diabetes Status",
        xlab = "Diabetes", ylab = "HbA1c Level")

boxplot(blood_glucose_level ~ diabetes, data = data,
        main = "Blood Glucose Level by Diabetes Status",
        xlab = "Diabetes", ylab = "Blood Glucose Level")

# Correlations with diabetes coded as numeric 0/1
diabetes_num <- as.numeric(as.character(data$diabetes))

cor(data$age, diabetes_num)
cor(data$bmi, diabetes_num)
cor(data$HbA1c_level, diabetes_num)
cor(data$blood_glucose_level, diabetes_num)

############################################################
# part 2: Methods: KNN, DT and LR Classifiers
############################################################

#############
#data preparation
##############

# response variable
data$diabetes <- factor(data$diabetes, levels = c(0, 1))

# other categorical variables
data$gender <- factor(data$gender)
data$hypertension <- factor(data$hypertension, levels = c(0, 1))
data$heart_disease <- factor(data$heart_disease, levels = c(0, 1))

# check result
str(data)
table(data$smoking_history)
sum(is.na(data$smoking_history))

######################
# helper functions
#######################

# to get metrics from predicted classes
get_metrics <- function(actual, predicted) {
  actual <- factor(actual, levels = c("0", "1"))
  predicted <- factor(predicted, levels = c("0", "1"))
  
  cm <- table(Predicted = predicted, Actual = actual)
  
  TN <- cm["0", "0"]
  FN <- cm["0", "1"]
  FP <- cm["1", "0"]
  TP <- cm["1", "1"]
  
  accuracy  <- (TP + TN) / (TP + TN + FP + FN)
  tpr       <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))   # sensitivity
  fpr       <- ifelse((FP + TN) == 0, NA, FP / (FP + TN))
  fnr       <- ifelse((FN + TP) == 0, NA, FN / (FN + TP))
  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  
  return(c(Accuracy = accuracy,
           TPR = tpr,
           FPR = fpr,
           FNR = fnr,
           Precision = precision))
}

# to prepare KNN design matrices
prepare_knn_data <- function(train_df, test_df) {
  
  x_train <- model.matrix(diabetes ~ gender + age + hypertension + heart_disease +
                            smoking_history + bmi + HbA1c_level + blood_glucose_level,
                          data = train_df)[, -1]
  
  x_test <- model.matrix(diabetes ~ gender + age + hypertension + heart_disease +
                           smoking_history + bmi + HbA1c_level + blood_glucose_level,
                         data = test_df)[, -1]
  
  y_train <- train_df$diabetes
  y_test  <- test_df$diabetes
  
  # remove zero-variance columns based on training data
  keep_cols <- apply(x_train, 2, sd) != 0
  
  x_train <- x_train[, keep_cols, drop = FALSE]
  x_test  <- x_test[, keep_cols, drop = FALSE]
  
  # scale using training set only
  x_train_scaled <- scale(x_train)
  
  x_test_scaled <- scale(x_test,
                         center = attr(x_train_scaled, "scaled:center"),
                         scale  = attr(x_train_scaled, "scaled:scale"))
  
  return(list(
    x_train = x_train_scaled,
    x_test  = x_test_scaled,
    y_train = y_train,
    y_test  = y_test
  ))
}

############################################################
# find best version of each method on 10,000 sample
############################################################

# a random sample of 10,000 for tuning
set.seed(304)
idx_10k <- sample(1:nrow(data), 10000)
subdata <- data[idx_10k, ]

# same 80/20 split for all methods in Step 4
set.seed(304)
train_idx_sub <- sample(1:nrow(subdata), size = 0.8 * nrow(subdata))

train_sub <- subdata[train_idx_sub, ]
test_sub  <- subdata[-train_idx_sub, ]

############################################################
# A. KNN tuning
############################################################

knn_data_sub <- prepare_knn_data(train_sub, test_sub)

k_values <- 1:20

knn_results <- data.frame(
  k = k_values,
  Accuracy = numeric(length(k_values)),
  TPR = numeric(length(k_values)),
  FPR = numeric(length(k_values)),
  FNR = numeric(length(k_values)),
  Precision = numeric(length(k_values))
)

for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  pred <- knn(train = knn_data_sub$x_train,
              test  = knn_data_sub$x_test,
              cl    = knn_data_sub$y_train,
              k     = k)
  
  m <- get_metrics(knn_data_sub$y_test, pred)
  
  knn_results$Accuracy[i]  <- m["Accuracy"]
  knn_results$TPR[i]       <- m["TPR"]
  knn_results$FPR[i]       <- m["FPR"]
  knn_results$FNR[i]       <- m["FNR"]
  knn_results$Precision[i] <- m["Precision"]
}

knn_results

best_k <- knn_results$k[which.max(knn_results$Accuracy)]
best_k

# plot KNN performance
matplot(knn_results$k,
        knn_results[, c("Accuracy", "Precision", "TPR", "FNR", "FPR")],
        type = "b",
        pch = 16,
        lty = 1,
        lwd = 2,
        col = c("blue", "purple", "green", "orange", "red"),
        xlab = "k",
        ylab = "Metrics",
        main = "KNN Performance by k Value",
        ylim = c(0, 1))

abline(v = best_k, lty = 2, col = "grey60")
text(best_k + 0.3, 0.5, paste("k =", best_k), cex = 0.9)

legend("bottomright",
       legend = c("Accuracy", "Precision", "TPR", "FNR", "FPR"),
       col = c("blue", "purple", "green", "orange", "red"),
       lty = 1,
       pch = 16,
       bty = "n")

############################################################
# B. Decision Tree tuning
############################################################

cp_values <- 10^(-(1:10))

dt_results <- data.frame(
  cp = cp_values,
  Accuracy = numeric(length(cp_values)),
  TPR = numeric(length(cp_values)),
  FPR = numeric(length(cp_values)),
  FNR = numeric(length(cp_values)),
  Precision = numeric(length(cp_values))
)

for (i in seq_along(cp_values)) {
  cp_i <- cp_values[i]
  
  dt_model <- rpart(
    diabetes ~ gender + age + hypertension + heart_disease +
      smoking_history + bmi + HbA1c_level + blood_glucose_level,
    data = train_sub,
    method = "class",
    control = rpart.control(cp = cp_i)
  )
  
  dt_prob <- predict(dt_model, newdata = test_sub, type = "prob")[, "1"]
  dt_pred <- ifelse(dt_prob >= 0.5, "1", "0")
  dt_pred <- factor(dt_pred, levels = c("0", "1"))
  
  m <- get_metrics(test_sub$diabetes, dt_pred)
  
  dt_results$Accuracy[i]  <- m["Accuracy"]
  dt_results$TPR[i]       <- m["TPR"]
  dt_results$FPR[i]       <- m["FPR"]
  dt_results$FNR[i]       <- m["FNR"]
  dt_results$Precision[i] <- m["Precision"]
}

dt_results

best_cp <- dt_results$cp[which.max(dt_results$Accuracy)]
best_cp

# plot DT performance
matplot(1:10,
        dt_results[, c("Accuracy", "Precision", "TPR", "FNR", "FPR")],
        type = "b",
        pch = 16,
        lty = 1,
        lwd = 2,
        col = c("blue", "purple", "green", "orange", "red"),
        xaxt = "n",
        xlab = "cp value",
        ylab = "Metrics",
        main = "Decision Tree Performance by cp Value",
        ylim = c(0, 1))

axis(1, at = 1:10, labels = paste0("1e-", 1:10))

legend("bottomright",
       legend = c("Accuracy", "Precision", "TPR", "FNR", "FPR"),
       col = c("blue", "purple", "green", "orange", "red"),
       lty = 1,
       pch = 16,
       bty = "n")

############################################################
# C. Logistic Regression model selection
############################################################

# model 1: Full model
lr_model1 <- glm(
  diabetes ~ gender + age + hypertension + heart_disease +
    smoking_history + bmi + HbA1c_level + blood_glucose_level,
  data = train_sub,
  family = binomial
)

lr_prob1 <- predict(lr_model1, newdata = test_sub, type = "response")
lr_pred1 <- ifelse(lr_prob1 >= 0.5, "1", "0")
lr_pred1 <- factor(lr_pred1, levels = c("0", "1"))
lr_m1 <- get_metrics(test_sub$diabetes, lr_pred1)

# model 2: Reduced model without gender and smoking_history
lr_model2 <- glm(
  diabetes ~ age + hypertension + heart_disease +
    bmi + HbA1c_level + blood_glucose_level,
  data = train_sub,
  family = binomial
)

lr_prob2 <- predict(lr_model2, newdata = test_sub, type = "response")
lr_pred2 <- ifelse(lr_prob2 >= 0.5, "1", "0")
lr_pred2 <- factor(lr_pred2, levels = c("0", "1"))
lr_m2 <- get_metrics(test_sub$diabetes, lr_pred2)

# model 3: Quantitative-only model
lr_model3 <- glm(
  diabetes ~ age + bmi + HbA1c_level + blood_glucose_level,
  data = train_sub,
  family = binomial
)

lr_prob3 <- predict(lr_model3, newdata = test_sub, type = "response")
lr_pred3 <- ifelse(lr_prob3 >= 0.5, "1", "0")
lr_pred3 <- factor(lr_pred3, levels = c("0", "1"))
lr_m3 <- get_metrics(test_sub$diabetes, lr_pred3)

lr_results <- data.frame(
  Model = c("Full", "Reduced", "Quantitative only"),
  Accuracy = c(lr_m1["Accuracy"], lr_m2["Accuracy"], lr_m3["Accuracy"]),
  TPR = c(lr_m1["TPR"], lr_m2["TPR"], lr_m3["TPR"]),
  FPR = c(lr_m1["FPR"], lr_m2["FPR"], lr_m3["FPR"]),
  FNR = c(lr_m1["FNR"], lr_m2["FNR"], lr_m3["FNR"]),
  Precision = c(lr_m1["Precision"], lr_m2["Precision"], lr_m3["Precision"])
)

lr_results

best_lr_name <- lr_results$Model[which.max(lr_results$Accuracy)]
best_lr_name

# pick best LR model object
if (best_lr_name == "Full") {
  best_lr_formula <- diabetes ~ gender + age + hypertension + heart_disease +
    smoking_history + bmi + HbA1c_level + blood_glucose_level
  best_lr_model_sub <- lr_model1
}

if (best_lr_name == "Reduced") {
  best_lr_formula <- diabetes ~ age + hypertension + heart_disease +
    bmi + HbA1c_level + blood_glucose_level
  best_lr_model_sub <- lr_model2
}

if (best_lr_name == "Quantitative only") {
  best_lr_formula <- diabetes ~ age + bmi + HbA1c_level + blood_glucose_level
  best_lr_model_sub <- lr_model3
}

summary(best_lr_model_sub)
round(as.data.frame(summary(best_lr_model_sub)$coefficients), 4)
round(coef(best_lr_model_sub), 4)

############################################################
# final comparison on full data
############################################################

# same 80/20 split for all 3 methods on full data
set.seed(304)
train_idx_full <- sample(1:nrow(data), size = 0.8 * nrow(data))

train_full <- data[train_idx_full, ]
test_full  <- data[-train_idx_full, ]

############################################################
# A. Final KNN on full data using best_k
############################################################

knn_data_full <- prepare_knn_data(train_full, test_full)

knn_pred_full <- knn(train = knn_data_full$x_train,
                     test  = knn_data_full$x_test,
                     cl    = knn_data_full$y_train,
                     k     = best_k,
                     prob  = TRUE)

knn_metrics_full <- get_metrics(knn_data_full$y_test, knn_pred_full)

# approximate class-1 probability for ROC/AUC
knn_vote_prop <- attr(knn_pred_full, "prob")
knn_prob_full <- ifelse(knn_pred_full == "1", knn_vote_prop, 1 - knn_vote_prop)

knn_roc_full <- roc(response = knn_data_full$y_test,
                    predictor = knn_prob_full,
                    levels = c("0", "1"))

knn_auc_full <- auc(knn_roc_full)

############################################################
# B. Final Decision Tree on full data using best_cp
############################################################

dt_final <- rpart(
  diabetes ~ gender + age + hypertension + heart_disease +
    smoking_history + bmi + HbA1c_level + blood_glucose_level,
  data = train_full,
  method = "class",
  control = rpart.control(cp = best_cp)
)

dt_prob_full <- predict(dt_final, newdata = test_full, type = "prob")[, "1"]
dt_pred_full <- ifelse(dt_prob_full >= 0.5, "1", "0")
dt_pred_full <- factor(dt_pred_full, levels = c("0", "1"))

dt_metrics_full <- get_metrics(test_full$diabetes, dt_pred_full)

dt_roc_full <- roc(response = test_full$diabetes,
                   predictor = dt_prob_full,
                   levels = c("0", "1"))

dt_auc_full <- auc(dt_roc_full)

############################################################
# C. Final Logistic Regression on full data using best formula
############################################################

lr_final <- glm(
  formula = best_lr_formula,
  data = train_full,
  family = binomial
)

summary(lr_final)
round(as.data.frame(summary(lr_final)$coefficients), 4)
round(coef(lr_final), 4)

lr_prob_full <- predict(lr_final, newdata = test_full, type = "response")
lr_pred_full <- ifelse(lr_prob_full >= 0.5, "1", "0")
lr_pred_full <- factor(lr_pred_full, levels = c("0", "1"))

lr_metrics_full <- get_metrics(test_full$diabetes, lr_pred_full)

lr_roc_full <- roc(response = test_full$diabetes,
                   predictor = lr_prob_full,
                   levels = c("0", "1"))

lr_auc_full <- auc(lr_roc_full)

############################################################
# final comparison tables
############################################################

comparison_table <- data.frame(
  Method = c("KNN", "Decision Tree", "Logistic Regression"),
  Accuracy = c(knn_metrics_full["Accuracy"], dt_metrics_full["Accuracy"], lr_metrics_full["Accuracy"]),
  TPR = c(knn_metrics_full["TPR"], dt_metrics_full["TPR"], lr_metrics_full["TPR"]),
  AUC = c(as.numeric(knn_auc_full), as.numeric(dt_auc_full), as.numeric(lr_auc_full))
)

comparison_table

comparison_table_full <- data.frame(
  Method = c("KNN", "Decision Tree", "Logistic Regression"),
  Accuracy = c(knn_metrics_full["Accuracy"], dt_metrics_full["Accuracy"], lr_metrics_full["Accuracy"]),
  TPR = c(knn_metrics_full["TPR"], dt_metrics_full["TPR"], lr_metrics_full["TPR"]),
  FPR = c(knn_metrics_full["FPR"], dt_metrics_full["FPR"], lr_metrics_full["FPR"]),
  FNR = c(knn_metrics_full["FNR"], dt_metrics_full["FNR"], lr_metrics_full["FNR"]),
  Precision = c(knn_metrics_full["Precision"], dt_metrics_full["Precision"], lr_metrics_full["Precision"]),
  AUC = c(as.numeric(knn_auc_full), as.numeric(dt_auc_full), as.numeric(lr_auc_full))
)

comparison_table_full

############################################################
# plot ROC curves together
############################################################

plot(knn_roc_full, col = "blue", lwd = 2,
     main = "ROC Curves for Final Models")
plot(dt_roc_full, col = "red", lwd = 2, add = TRUE)
plot(lr_roc_full, col = "darkgreen", lwd = 2, add = TRUE)

legend("bottomright",
       legend = c(
         paste0("KNN (AUC = ", round(knn_auc_full, 4), ")"),
         paste0("DT (AUC = ", round(dt_auc_full, 4), ")"),
         paste0("LR (AUC = ", round(lr_auc_full, 4), ")")
       ),
       col = c("blue", "red", "darkgreen"),
       lwd = 2,
       bty = "n")

