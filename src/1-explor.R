# Torre Analytics test
## Exploration of the dataset given


# 0.0 LIBS AND FUNS----
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)
library(stringr)
library(purrr)
library(forcats)
library(tidytext)
library(patchwork)
library(ggsci)

top_chars <- function(col, n=12){
  
  torre %>% count(!!sym(col)) %>% arrange(desc(n)) %>% head(n) %>% pull(!!sym(col))
  
}

# 1.0 READ DATA AND FIRST IMPRESSIONS ----

torre <- read_csv("data/reed_uk.csv") %>% 
  mutate(city=fct_infreq(factor(str_to_title(city))), category=fct_infreq(factor(category)),
         post_date=mdy(post_date), month=month(post_date), weekday=wday(post_date))

# 1.1 Summary  ----

glimpse(torre)
summary(torre)

# 2.0 EDA ----

# 2.1 Unique and Empty Values ----

empty <-map_dbl(setNames(1:ncol(torre), names(torre)), ~sum(is.na(torre[,.x])))

round(empty/nrow(torre), 2)

# Job board and geo must be deleted
uniq_vals <- map_dbl(setNames(1:ncol(torre), names(torre)), ~length(unique(torre[[.x]])))

# Only the first 3 months have a lot of data
torre %>% count(month(post_date)) %>% arrange(desc(n))

# The most feasible predictor is job category, which only has 9 levels
# The EDA will continue by putting this variables at the front

# 2.2 Graphical Exploration ----

# Geography
top_city <- top_chars("city")

p1 <- torre %>% 
  filter(city %in% top_city) %>% 
  count(city, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~city, scales = "free_x") + coord_flip() +
  labs(y="Count", x="Job Type") +
  theme_bw() +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))

ggsave("graphs/eda/top_city.png", plot = p1, device = "png", units = "cm", width = 30, height = 30)

top_state <- top_chars("state")

p2 <- torre %>% 
  filter(state %in% top_state) %>% 
  count(state, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~state, scales = "free_x") + coord_flip() +
  labs(y="Count", x="Job Type") +
  theme_bw() +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))

ggsave("graphs/eda/top_state.png", plot = p2, device = "png", units = "cm", width = 30, height = 30)

# Job Classification
top_category <- top_chars("category")

p3 <- torre %>% 
  filter(category %in% top_category) %>% 
  count(category, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~category, scales = "free_x") + coord_flip() +
  labs(y="Count", x="Job Type") +
  theme_bw() +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))

ggsave("graphs/eda/top_category.png", plot = p3, device = "png", units = "cm", width = 30, height = 30)


# Temporal
p4 <- torre %>% 
  filter(month<=3) %>% # There is not much data outside these months
  count(month, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~month) + coord_flip() +
  labs(y="Count", x="Job Type") +
  theme_bw() +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))

ggsave("graphs/eda/month.png", plot = p4, device = "png", units = "cm", width = 30, height = 30)

p5 <- torre %>%
  mutate(w=wday(post_date, label = TRUE)) %>% 
  count(w, job_type) %>% 
  ggplot(aes(job_type, n)) + geom_col() +
  facet_wrap(~w) + coord_flip() +
  labs(y="Count", x="Job Type") +
  theme_bw() +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))

ggsave("graphs/eda/weekday.png", plot = p5, device = "png", units = "cm", width = 30, height = 30)


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

p6 <- count_words %>% 
  ggplot(aes(reorder_within(word, n, job_type), n, fill=job_type)) + geom_col() +
  facet_wrap(~job_type, scales="free")  +
  coord_flip() +
  scale_x_reordered() +
  guides(fill=FALSE) +
  theme_bw() +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold"))

ggsave("graphs/eda/keywords.png", plot = p6, device = "png", units = "cm", width = 30, height = 30)

# Posible predictors: city, state, category, week, month, description

# Description needs more preprocessing. Due to its complex nature,
# it can be process with more tools and better resources that currently 
# available





