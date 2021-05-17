# Torre Analytics test
## Prediction based models of job_type

# 0.0 LIBS AND READ
library(readr)
library(recipes)
library(tensorflow)
library(embed) 
library(reticulate)
library(caret)
library(rsample)
library(doParallel)
library(Metrics)
library(tfruns)
library(stringr)
library(forcats)
library(lubridate)

source("src/funs/df_to_dataset.R")

load("data/train_obj/data_baked.RData")
load("data/train_obj/data_deep.RData")


# 1.0 CARET MODELLING ----

# 1,1 Control and Grids ----
control <- trainControl(method  = "cv", number  = 5)

ridge_grid <- expand.grid(alpha  = 0, 
                          lambda = 10^seq(10,-2, length = 50))

lasso_grid <- expand.grid(alpha = 1,
                          lambda = 10^seq(10,-2, length = 50))

tree_grid  <- expand.grid(maxdepth = seq(5, 30, 5))

rf_grid    <- expand.grid(mtry = seq(2, 5, 1))

knn_grid <- expand.grid(k=seq(5, 50, 5))


# 1.2 Training ----
cores <- detectCores()
cl <- makePSOCKcluster(cores-2)

registerDoParallel(cl)

set.seed(2001)

tictoc::tic()
ridge_mod <- train(job_type ~ ., data = train_bake, 
                   trControl = control, method = "glmnet",
                   tuneGrid  = ridge_grid, family = "multinomial", 
                   type.multinomial = "grouped")

lasso_mod <- train(job_type ~ ., data = train_bake,
                   trControl = control, method = "glmnet",
                   tuneGrid  = lasso_grid, family = "multinomial", 
                   type.multinomial = "grouped")

# tree_mod  <- train(job_type ~ ., data = train_bake,
#                    trControl = control, method = "rpart2",
#                    tuneGrid  = tree_grid)

rf_mod    <- train(job_type ~ ., data = train_bake,
                   trControl = control, method = "rf",
                   tuneGrid  = rf_grid)

knn_mod   <- train(job_type ~ ., data = train_bake, 
                   trControl = control, method = "knn",
                   tuneGrid  = knn_grid)

stopCluster(cl)
tictoc::toc()

save(ridge_mod, lasso_mod, rf_mod, knn_mod, file = "mods/mods_caret.RData")

# 3.0 TF MODELLING ----

# 3.1 Random Grid Search ----
runs <- tuning_run("src/nnet-grid.R", 
                   flags = list(
                     nodes1 = c(32, 64, 128),
                     nodes2 = c(32, 64, 128),
                     nodes3 = c(32, 64, 128),
                     nodes4 = c(32, 64, 128),
                     reg1 = c(0.25, 0.5, 0.75),
                     reg2 = c(0.25, 0.5, 0.75),
                     lrannea1 = c(0.001, 0.01, 0.1),
                     lrannea2 = c(0.1, 0.05)
                   ),
                   sample = 0.01
)

# 3.2 Best Hyperparameter Settings ----
best_r <- ls_runs() %>% arrange(desc(metric_val_accuracy)) %>% filter(str_detect(run_dir, "^runs/2021"))%>% 
  select(starts_with("flag"), metric_accuracy, metric_val_accuracy) %>% 
  head(2)


# 3.3 Model with Best Settings ----
batch_size <- 700 # nrow(train)/50 # 700
train_ds   <- df_to_dataset(train_deep, batch_size = batch_size)
eval_ds    <- df_to_dataset(eval_deep, shuffle = FALSE, batch_size = batch_size)
test_ds    <- df_to_dataset(test_deep, shuffle = FALSE, batch_size = batch_size)

spec <- feature_spec(train_ds, job_type ~ .) %>% 
  step_numeric_column(all_numeric())

spec_prep <- fit(spec)

nn_mod <- keras_model_sequential() %>% 
  layer_dense_features(dense_features(spec_prep)) %>% 
  layer_dense(units = best_r$flag_nodes1[1], activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = best_r$flag_nodes2[1], activation = "relu", 
              kernel_regularizer = regularizer_l2(l = best_r$flag_reg1[1])) %>%
  layer_batch_normalization() %>%
  layer_dense(units = best_r$flag_nodes3[1], activation = "relu", 
              kernel_regularizer = regularizer_l2(l = best_r$flag_reg2[1])) %>%
  layer_batch_normalization() %>%
  layer_dense(units = best_r$flag_nodes4[1], activation = "relu") %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 9, activation = "softmax") %>% 
  compile(
    loss = loss_sparse_categorical_crossentropy, 
    optimizer = optimizer_adam(lr = best_r$flag_lrannea1[1]), 
    metrics = "accuracy"
  ) 


hist_nn <- nn_mod %>% 
  fit(
    dataset_use_spec(train_ds, spec = spec_prep),
    epochs = 50, 
    callbacks = list(callback_reduce_lr_on_plateau(factor = best_r$flag_lrannea2[1])),
    validation_data = dataset_use_spec(eval_ds, spec_prep),
    verbose = 2
  )

save_model_tf(nn_mod, "mods/nnet/")

# 4.0 AUC RESULTS ----

pred_ridge <- predict(ridge_mod, newdata = test_bake %>% select(-job_type))
pred_lasso <- predict(lasso_mod, newdata = test_bake %>% select(-job_type))
pred_rf    <- predict(rf_mod, newdata = test_bake %>% select(-job_type))
pred_knn   <- predict(knn_mod, newdata = test_bake %>% select(-job_type))
pred_nn    <- predict_classes(nn_mod, test_deep %>% select(-job_type))

pred_res <- c(ridge = multiclass.roc(response=test_bake$job_type, predictor=as.numeric(pred_ridge))$auc,
              lasso = multiclass.roc(response=test_bake$job_type, predictor=as.numeric(pred_lasso))$auc,
              rf    = multiclass.roc(response=test_bake$job_type, predictor=as.numeric(pred_rf))$auc,
              knn   = multiclass.roc(response=test_bake$job_type, predictor=as.numeric(pred_knn))$auc,
              nn    = multiclass.roc(response=test_deep$job_type, predictor=as.numeric(pred_nn))$auc)

preds <- tibble(ridge = pred_ridge, lasso=pred_lasso, rf=pred_rf, knn=pred_knn, 
                real=test_bake$job_type)

preds_nn <- tibble(nn=pred_nn, real=test_deep$job_type)

save(pred_res, preds, preds_nn, file = "res/pred.RData")

