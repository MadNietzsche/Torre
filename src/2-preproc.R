library(recipes)
library(tensorflow)
library(embed) 
library(reticulate)
library(caret)
library(rsample)
library(doParallel)

set.seed(2001)

splits   <- initial_split(torre, prop = 0.7)
train_df <- training(splits)
test_df  <- testing(splits)

torre_rec <- recipe(job_type ~ city + state + category + weekday + month + job_description, data = torre) %>% 
  step_mutate(commas=str_count(job_description, ","), exclam=str_count(job_description, "!"),
              money=str_count(job_description, "£|$"), dash=str_count(job_description, "-"), 
              colon=str_count(job_description, ":"), parenthesis=str_count(job_description, "\\(")) %>% 
  step_embed(
    city, state, category,
    outcome = vars(job_type),
    num_terms = 5,
    hidden_units = 30,
    predictors = NULL,
    options = embed_control(
      loss = "binary_crossentropy",
      epochs = 10,
      validation_split = 0.2,
      verbose = 0
    )
  ) %>% 
  step_rm(job_description) %>% 
  step_zv(all_predictors()) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors()) %>% 
  prep()

train_bake <- bake(torre_rec, train_df)
test_bake  <- bake(torre_rec, test_df)

control <- trainControl(method  = "cv", number  = 5)

ridge_grid <- expand.grid(alpha  = 0, 
                          lambda = 10^seq(10,-2, length = 50))

lasso_grid <- expand.grid(alpha = 1,
                          lambda = 10^seq(10,-2, length = 50))

tree_grid  <- expand.grid(maxdepth = seq(5, 30, 5))

rf_grid    <- expand.grid(mtry = seq(2, 5, 1))

knn_grid <- expand.grid(k=seq(5, 50, 5))


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





