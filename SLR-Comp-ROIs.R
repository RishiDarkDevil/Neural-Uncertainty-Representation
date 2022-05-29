library(ggpubr)
library(viridis)

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

# Visualizae ROI Regression
vis_des_on_asc <- function(data, ROI_name, mods = NULL) {
  data <- data %>%
    filter(ROI == ROI_name)
  
  plots <- list()
  for (i in seq_along(asc_TRs)) {
    data_model <- data %>%
      filter(phase %in% str_c(c("A", "D"), i))
    
    data_model <- data_model %>%
      spread(phase, BOLD_mean)

    plots[[i]] <- data_model %>%
      ggplot(aes(data_model[[str_c("A",i)]], data_model[[str_c("D",i)]])) +
      geom_point(size = 1.5) +
      geom_smooth(method = "lm", formula = y~x, se = FALSE, size = 2.5) +
      labs(
        x = str_c("A",i),
        y = str_c("D",i)
      ) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),strip.background = element_blank()) +
      theme(
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        strip.background = element_blank()
      ) + 
      scale_color_viridis(discrete = TRUE)
  }
  if (!is.null(mods)) {
    plots[[i+1]] <- present_R.2(mods)
  }
  
  return(plots)
}

# Converting the data into long form for easier modeling
model_data <- BOLD_data_avg %>%
  gather("ROI", "BOLD_mean", -ID, -phase)

# Fitting the BOLD Descent on Ascent for the different ROIs
ROIs <- str_c(c("VC", "l_OFC", "m_OFC"), "_1")

model_ROIs <- ROIs %>%
  map(~model_des_on_asc(model_data, .))

# Visualize the BOLD Descent on Ascent for the different ROIs
plot_ROIs <- list()
for (i in seq_along(ROIs)) {
  plot_ROIs[[i]] <- vis_des_on_asc(model_data, ROIs[i], model_ROIs[[i]])
}

# VC Visualization
plot_VC <- ggarrange(plotlist = plot_ROIs[[1]], ncol = 3, nrow = 2)
annotate_figure(plot_VC, top = text_grob("VC BOLD Descent on Ascent", face = "bold", size = 16))

# l_OFC Visualization
plot_l_OFC <- ggarrange(plotlist = plot_ROIs[[2]], ncol = 3, nrow = 2)
annotate_figure(plot_l_OFC, top = text_grob("l_OFC BOLD Descent on Ascent", face = "bold", size = 16))

# m_OFC Visualization
plot_m_OFC <- ggarrange(plotlist = plot_ROIs[[3]], ncol = 3, nrow = 2)
annotate_figure(plot_m_OFC, top = text_grob("m_OFC BOLD Descent on Ascent", face = "bold", size = 16))
