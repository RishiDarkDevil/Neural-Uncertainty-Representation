# Splits data by a column and adds different labels to the splitted rows, returns the final labeled and split data 
split_label <- function(data, label_values, split_by_col) {
  # split_by_col contains the column whose values are labeled according to the values in label_values
  split_data <- list()
  for (i in names(label_values)) {
    split_data[[i]] <- data %>%
      filter(data[[split_by_col]] %in% label_values[[i]]) %>%
      mutate(phase = i)
  }
  split_data <- bind_rows(split_data)
  return(split_data)
}

# Example:
labels_and_values <- list("A1" = 57:68, "A2" = 74:82, "A3" = 135:143, "A4" = 156:168, "A5" = 175:181)
split_data <- split_label(data, labels_and_values, "column name")
