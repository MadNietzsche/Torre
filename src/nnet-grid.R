library(keras)
library(dplyr)
library(tfdatasets)

load("data/train_obj/data_deep.RData")
source("src/funs/df_to_dataset.R")

batch_size <- 700 # nrow(train)/50 # 700
train_ds   <- df_to_dataset(train_deep, batch_size = batch_size)
eval_ds    <- df_to_dataset(eval_deep, shuffle = FALSE, batch_size = batch_size)
test_ds    <- df_to_dataset(test_deep, shuffle = FALSE, batch_size = batch_size)

#save(train_ds, test_ds, eval_ds, file="data/train_obj/tensor_deep.RData")

spec <- feature_spec(train_ds, job_type ~ .) %>% 
  step_numeric_column(all_numeric())

spec_prep <- fit(spec)

FLAGS <- tfruns::flags(
  # nodes
  flag_numeric("nodes1", 32),
  flag_numeric("nodes2", 64),
  flag_numeric("nodes3", 64),
  flag_numeric("nodes4", 32),
  # dropout
  flag_numeric("reg1", 0.5),
  flag_numeric("reg2", 0.3),
  flag_numeric("lrannea1", 0.001),
  flag_numeric("lrannea2", 0.1)
)

model <- keras_model_sequential() %>% 
  layer_dense_features(dense_features(spec_prep)) %>% 
  layer_dense(units = FLAGS$nodes1, activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = FLAGS$nodes2, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = FLAGS$reg1)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = FLAGS$nodes3, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = FLAGS$reg2)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = FLAGS$nodes4, activation = "relu") %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 9, activation = "softmax") %>% 
  compile(
    loss = loss_sparse_categorical_crossentropy, 
    optimizer = optimizer_adam(lr = FLAGS$lrannea1), 
    metrics = "accuracy"
  ) %>% 
  fit(
    dataset_use_spec(train_ds, spec = spec_prep),
    epochs = 50, 
    callbacks = list(callback_early_stopping(patience = 7),
                     callback_reduce_lr_on_plateau(factor = FLAGS$lrannea2)),
    validation_data = dataset_use_spec(eval_ds, spec_prep),
    verbose = 2
  )

