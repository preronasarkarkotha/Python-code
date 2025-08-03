install.packages("glmnet")
install.packages("randomForest")
install.packages("xgboost")
install.packages("adabag")
install.packages("caretEnsemble")
install.packages("pROC")
install.packages("readr")
install.packages("tm")
install.packages("caret")
install.packages("text2vec")
install.packages("SnowballC")

# Install packages if not already installed (run once)
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) install.packages(new_packages)
}
install_if_missing(c("readr", "dplyr", "tm", "caret", "text2vec", "glmnet", "randomForest", "xgboost", "adabag", "caretEnsemble", "pROC"))

library(readr)
library(tm)
library(caret)
library(text2vec)
library(glmnet)
library(randomForest)
library(xgboost)
library(adabag)
library(caretEnsemble)
library(pROC)

library(readr)
library(dplyr)

zip_path <- "C:/Users/himel/Downloads/News-_dataset.zip"
files_in_zip <- unzip(zip_path, list = TRUE)
print(files_in_zip)


temp_folder <- tempdir()        # Temporary folder path
unzip(zip_path, exdir = temp_folder)

# 4. List all CSV files in the extracted folder
csv_files <- list.files(temp_folder, pattern = "\\.csv$", full.names = TRUE)
print(paste("CSV files found:", paste(csv_files, collapse = ", ")))

# --- 4. Read each CSV separately and add label ---
fake_df <- read_csv(csv_files[grep("Fake.csv", csv_files)], show_col_types = FALSE) %>%
  mutate(label = factor(0, levels = c(0, 1)))

true_df <- read_csv(csv_files[grep("True.csv", csv_files)], show_col_types = FALSE) %>%
  mutate(label = factor(1, levels = c(0, 1)))
# Check combined data
print(head(combined_data))
print(paste("Total rows combined:", nrow(combined_data)))

# --- 5. Combine fake and true news datasets ---
combined_data <- bind_rows(fake_df, true_df)

# Sanity check
print(table(combined_data$label))
print(head(combined_data))

# --- 6. Define text cleaning function ---
clean_text <- function(text) {
  corpus <- VCorpus(VectorSource(text))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  sapply(corpus, as.character)
}
# --- 7. Clean the 'text' column ---
combined_data$clean_text <- clean_text(combined_data$text)

# --- 8. Split into train/test datasets ---
set.seed(123)
train_indices <- createDataPartition(combined_data$label, p = 0.8, list = FALSE)
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# --- 7. Create itoken iterators ---
it_train <- itoken(train_data$clean_text, progressbar = FALSE)
it_test <- itoken(test_data$clean_text, progressbar = FALSE)

# --- 8a. Full Vocabulary TF-IDF (for glmnet and xgboost) ---
vocab_full <- create_vocabulary(it_train)
vectorizer_full <- vocab_vectorizer(vocab_full)

dtm_train_full <- create_dtm(it_train, vectorizer_full)
dtm_test_full <- create_dtm(it_test, vectorizer_full)

tfidf_full <- TfIdf$new()
dtm_train_tfidf_full <- tfidf_full$fit_transform(dtm_train_full)
dtm_test_tfidf_full <- tfidf_full$transform(dtm_test_full)

# --- 8b. Reduced Vocabulary TF-IDF (top 1000 terms) for RF and AdaBoost ---
vocab_pruned <- prune_vocabulary(vocab_full,
                                 term_count_min = 10,
                                 doc_proportion_min = 0.001,
                                 doc_proportion_max = 0.5)

top_n <- 1000
top_terms <- vocab_pruned$term[order(vocab_pruned$term_count, decreasing = TRUE)][1:min(top_n, nrow(vocab_pruned))]
vocab_reduced <- vocab_pruned[vocab_pruned$term %in% top_terms, ]
vectorizer_reduced <- vocab_vectorizer(vocab_reduced)

dtm_train_reduced <- create_dtm(it_train, vectorizer_reduced)
dtm_test_reduced <- create_dtm(it_test, vectorizer_reduced)

tfidf_reduced <- TfIdf$new()
dtm_train_tfidf_reduced <- tfidf_reduced$fit_transform(dtm_train_reduced)
dtm_test_tfidf_reduced <- tfidf_reduced$transform(dtm_test_reduced)

# Convert reduced TF-IDF to dense data frames for RF and AdaBoost
x_train_dense <- as.data.frame(as.matrix(dtm_train_tfidf_reduced))
x_test_dense <- as.data.frame(as.matrix(dtm_test_tfidf_reduced))

# Prepare labels
y_train <- train_data$label
y_test <- test_data$label

# --- 9a. Logistic Regression (glmnet) on full sparse ---
set.seed(123)
cv_glmnet <- cv.glmnet(dtm_train_tfidf_full, as.numeric(as.character(y_train)), 
                       family = "binomial", type.measure = "auc", nfolds = 5)
best_lambda <- cv_glmnet$lambda.min
glmnet_model <- glmnet(dtm_train_tfidf_full, as.numeric(as.character(y_train)), 
                       family = "binomial", lambda = best_lambda)
glmnet_pred_prob <- predict(glmnet_model, dtm_test_tfidf_full, type = "response")
glmnet_pred_label <- ifelse(glmnet_pred_prob > 0.5, 1, 0)

cat("\n--- Logistic Regression (glmnet) ---\n")
print(confusionMatrix(factor(glmnet_pred_label), y_test))
cat("AUC:", auc(y_test, as.vector(glmnet_pred_prob)), "\n\n")

# --- 9b. Random Forest on reduced dense ---
set.seed(123)
rf_model <- randomForest(x = x_train_dense, y = y_train, ntree = 100)
rf_pred <- predict(rf_model, x_test_dense)
rf_pred_prob <- as.numeric(rf_pred) - 1

cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, y_test))
cat("AUC:", auc(y_test, rf_pred_prob), "\n\n")

# --- 9c. XGBoost on full sparse ---
y_train_xgb <- as.numeric(as.character(y_train))
y_test_xgb <- as.numeric(as.character(y_test))
xgb_train <- xgb.DMatrix(data = dtm_train_tfidf_full, label = y_train_xgb)
xgb_test <- xgb.DMatrix(data = dtm_test_tfidf_full)

set.seed(123)
xgb_model <- xgboost(data = xgb_train, max.depth = 6, nrounds = 100,
                     objective = "binary:logistic", eval_metric = "auc", verbose = 0)

xgb_pred_prob <- predict(xgb_model, xgb_test)
xgb_pred_label <- ifelse(xgb_pred_prob > 0.5, 1, 0)

cat("\n--- XGBoost ---\n")
print(confusionMatrix(factor(xgb_pred_label), y_test))
cat("AUC:", auc(y_test_xgb, xgb_pred_prob), "\n\n")

# --- AdaBoost (adabag) on reduced dense ---
# Ensure features are numeric and colnames unique
colnames(x_train_dense) <- make.unique(make.names(colnames(x_train_dense)))
colnames(x_test_dense) <- make.unique(make.names(colnames(x_test_dense)))
x_train_dense[] <- lapply(x_train_dense, as.numeric)
x_test_dense[] <- lapply(x_test_dense, as.numeric)

dtm_train_df <- x_train_dense
dtm_test_df <- x_test_dense
dtm_train_df$label <- y_train
dtm_test_df$label <- y_test

set.seed(123)
ada_model <- boosting(label ~ ., data = dtm_train_df, boos = TRUE, mfinal = 50)
ada_pred <- predict(ada_model, newdata = dtm_test_df)
ada_pred_label <- ada_pred$class
ada_pred_prob <- ada_pred$prob[, 2]

cat("\n--- AdaBoost ---\n")
print(confusionMatrix(factor(ada_pred_label), y_test))
cat("AUC:", auc(as.numeric(as.character(y_test)), ada_pred_prob), "\n\n")


