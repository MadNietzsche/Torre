df_to_dataset <-
function(df, shuffle = TRUE, batch_size = 32) {
  ds <- df %>%
    tensor_slices_dataset()
  
  if (shuffle)
    ds <- ds %>% dataset_shuffle(buffer_size = nrow(df))
  
  ds %>%
    dataset_batch(batch_size = batch_size)
}
