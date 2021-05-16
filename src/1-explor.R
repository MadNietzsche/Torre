library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)
library(stringr)
library(purrr)
library(forcats)
library(tidytext)

top_chars <- function(col, n=12){
  
  torre %>% count(!!sym(col)) %>% arrange(desc(n)) %>% head(n) %>% pull(!!sym(col))
  
}

torre <- read_csv("data/reed_uk.csv") %>% 
  mutate(city=fct_infreq(factor(str_to_title(city))), category=fct_infreq(factor(category)),
         post_date=mdy(post_date), month=month(post_date), weekday=wday(post_date))

glimpse(torre)

# Empty cells

empty <-map_dbl(setNames(1:ncol(torre), names(torre)), ~sum(is.na(torre[,.x])))

round(empty/nrow(torre), 2)


# Job board and geo must be deleted
uniq_vals <- map_dbl(setNames(1:ncol(torre), names(torre)), ~length(unique(torre[[.x]])))

# Only the first 3 months have a lot of data
torre %>% count(month(post_date)) %>% arrange(desc(n))

# Geography
top_city <- top_chars("city")

torre %>% 
  filter(city %in% top_city) %>% 
  count(city, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~city, scales = "free_x") + coord_flip()

top_state <- top_chars("state")

torre %>% 
  filter(state %in% top_state) %>% 
  count(state, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~state, scales = "free_x") + coord_flip()

# Job Classification
top_category <- top_chars("category")

torre %>% 
  filter(category %in% top_category) %>% 
  count(category, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~category, scales = "free_x") + coord_flip()


# Temporal
torre %>% 
  filter(month<=3) %>% 
  count(month, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~mpnth) + coord_flip()

torre %>%
  mutate(w=wday(post_date, label = TRUE)) %>% 
  count(w, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~w) + coord_flip()



# Words
words_desc <- torre %>% 
  select(job_type, job_description) %>% 
  #group_by(job_type) %>% 
  unnest_tokens(word, job_description) %>% 
  anti_join(stop_words %>% 
              bind_rows(tibble(word=c("2", "3", "â", "II"), lexicon="OWN")))

count_words <- words_desc %>% 
  count(job_type, word) %>% 
  group_by(job_type) %>% 
  top_n(20, n) %>% ungroup()

count_words %>% 
  ggplot(aes(reorder_within(word, n, job_type), n, fill=job_type)) + geom_col() +
  facet_wrap(~job_type, scales="free")  +
  coord_flip() +
  scale_x_reordered() +
  guides(fill=FALSE)



# Posible predictors: city, state, category, week, month, description

# Description needs more preprocessing

words_more_50 <- words_desc %>%  
  count(word) %>% 
  filter(n>50)



