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
asc_TRs <- c(57:68, 74:82, 135:143, 156:168, 175:181)

# Descent TRs
des_TRs <- c(68:73, 85:90, 146:155, 169:175, 181:189)

# Ascent BOLD Data
BOLD_data_asc <- BOLD_data %>%
  filter(TR %in% asc_TRs)

# Descent BOLD Data
BOLD_data_des <- BOLD_data %>%
  filter(TR %in% des_TRs)
