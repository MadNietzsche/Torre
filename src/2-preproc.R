# Torre Analytics test
## Data Preprocessing before modelling

# 0.0 LIBS AND READ----
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

torre <- read_csv("data/reed_uk.csv") %>% 
  mutate(city=fct_infreq(factor(str_to_title(city))), category=fct_infreq(factor(category)),
         post_date=mdy(post_date), month=month(post_date), weekday=wday(post_date))

# 1.0 CARET PREPROCESSING ----

# 1.1 Splits ----

set.seed(2001)

splits   <- initial_split(torre, prop = 0.7)
train_df <- training(splits)
test_df  <- testing(splits)

# 1.1 Recipe

torre_rec <- recipe(job_type ~ city + state + category + weekday + month + job_description, data = torre) %>% 
  # Extract features from descripition
  step_mutate(commas=str_count(job_description, ","), exclam=str_count(job_description, "!"),
              money=str_count(job_description, "£|$"), dash=str_count(job_description, "-"), 
              colon=str_count(job_description, ":"), parenthesis=str_count(job_description, "\\(")) %>%
  # Create Embed Feature through NN
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
  # Remove Description and Zero Var cols
  step_rm(job_description) %>% 
  step_zv(all_predictors()) %>% 
  # Scaling
  step_center(all_predictors()) %>% 
  step_scale(all_predictors()) %>% 
  prep()

# 1.2 Baking ----
train_bake <- bake(torre_rec, train_df)
test_bake  <- bake(torre_rec, test_df)

# 2.0 TF PREPROCESSING ----

# 2.1 Splits ----
set.seed(2001)
train_ids <- sample(1:nrow(torre), size = 0.7*nrow(torre))
dev_ids <- split(setdiff(1:nrow(torre), train_ids),c(1,2))

# 2.2 Baking
train_deep <- bake(torre_rec, torre[train_ids,]) %>% mutate(job_type=as.numeric(job_type)-1)
eval_deep  <- bake(torre_rec, torre[dev_ids[[1]], ]) %>% mutate(job_type=as.numeric(job_type)-1)
test_deep  <- bake(torre_rec, torre[dev_ids[[2]], ]) %>% mutate(job_type=as.numeric(job_type)-1)

# 3.0 SAVING ----
save(train_bake, test_bake, file = "data/train_obj/data_baked.RData")
save(train_deep, eval_deep, test_deep, file = "data/train_obj/data_deep.RData")




