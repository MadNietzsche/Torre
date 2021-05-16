library(recipes)
library(tensorflow)
library(embed) 


torre_rec <- recipe(job_type ~ city + state + category + weekday + month + job_description, data = torre) %>% 
  step_mutate(commas=str_count(job_description, ","), exclam=str_count(job_description, "!"),
              money=str_count(job_description, "£|$"), dash=str_count(job_description, "-"), 
              colon=str_count(job_description, ":"), parenthesis=str_count(job_description, "\\(")) %>% 
  step_embed(
    city, state, category,
    outcome = vars(job_type),
    num_terms = 20,
    hidden_units = 30,
    predictors = NULL,
    options = embed_control(
      loss = "binary_crossentropy",
      epochs = 10,
      validation_split = 0.2,
      verbose = 0
    )
  ) %>% 
  step_rm(job_description, city, state, category) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors()) %>% 
  prep()


