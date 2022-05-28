# Averaging BOLD Data by ID and phase
BOLD_data_asc_avg <- BOLD_data_asc %>%
  select(-TR) %>%
  group_by(ID, phase) %>%
  summarise(across(everything(), list(mean)))

BOLD_data_des_avg <- BOLD_data_des %>%
  select(-TR) %>%
  group_by(ID, phase) %>%
  summarise(across(everything(), list(mean)))

BOLD_data_avg <- bind_rows(BOLD_data_asc_avg, BOLD_data_des_avg)

# Models all the Descents on Ascents by specified ROI
model_des_on_asc <- function(data, ROI_name) {
  data <- data %>%
    filter(ROI == ROI_name)
  
  models <- list()
  for (i in seq_along(asc_TRs)) {
    data_model <- data %>%
      filter(phase %in% str_c(c("A", "D"), i))
      
    data_model <- data_model %>%
      spread(phase, BOLD_mean)
  
    models[[i]] <- lm(as.formula(str_c("D",i,"~","A",i)), data = data_model)
  }
  
  return(models)
}

# Converting the data into long form for easier modeling
model_data <- BOLD_data_avg %>%
  gather("ROI", "BOLD_mean", -ID, -phase)

# Fitting the Descent on Ascent for the different ROIs
ROIs <- str_c(c("VC", "l_OFC", "m_OFC"), "_1")

model_ROIs <- ROIs %>%
  map(~model_des_on_asc(model_data, .))

model_ROIs[[3]] %>%
  map(~summary(.))
