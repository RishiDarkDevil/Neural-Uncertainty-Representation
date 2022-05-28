library(tidyverse)
library(readxl)

# Importing the BOLD Data from multiple sheets
BOLD_data <- excel_sheets("Whole_BOLD.xlsx") %>%
  map(~read_excel("Whole_BOLD.xlsx", sheet = .))

for (i in 1:length(BOLD_data)) {
  BOLD_data[[i]] <- BOLD_data[[i]] %>%
    mutate(ID = i, TR = row_number())
}

# Creating one single BOLD Data with Subjects indexed by IDs
BOLD_data <- bind_rows(BOLD_data) %>%
  select(ID, TR, everything())

# Ascent TRs
asc_TRs <- list("A1" = 57:68, "A2" = 74:82, "A3" = 135:143, "A4" = 156:168, "A5" = 175:181)

# Descent TRs
des_TRs <- list("D1" = 68:73, "D2" = 85:90, "D3" = 146:155, "D4" = 169:175, "D5" = 181:189)

# Ascent BOLD Data
BOLD_data_asc <- split_label(BOLD_data, asc_TRs, "TR")

# Descent BOLD Data
BOLD_data_des <- split_label(BOLD_data, des_TRs, "TR")
