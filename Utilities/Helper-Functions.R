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


# Present the R^2 when a list of models are passed that too visually - options allow numerical presentation too
present_R.2 <- function(mods, model_names = "", table_theme = "mBlue", footnote = "Model R-squared", visual = TRUE) {
  mods_summ <- mods %>%
    map(~summary(.))
  if(model_names == ""){
    model_names = 1:length(mods_summ)
  }
  R.squared <<- tibble("Model" = model_names,  
                       "R_square" = round(mods_summ %>%
                                            map_dbl(~.$r.squared), digits = 2), 
                       "Adj_R_square" = round(mods_summ %>%
                                                map_dbl(~.$adj.r.squared), digits = 2))
  if(!visual){
    p <- ggtexttable(R.squared, theme = ttheme(table_theme), rows = NULL) %>%
      tab_add_footnote(text = footnote, size = 10, face = "italic")}
  else{
    p <- R.squared %>%
      gather(R_square, Adj_R_square, key = "Metric", value = "value") %>%
      mutate(Model = as.factor(Model)) %>%
      ggplot(aes(fct_reorder(Model, value), value, color = Metric, group = Metric)) +
      geom_line(size = 1.5) +
      geom_point(size = 2.5) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),strip.background = element_blank()) +
      ggtitle("Model Performance") +
      theme(
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        strip.background = element_blank()
      ) + 
      labs(
        x = "Model",
        y = "Metric Value"
      ) + 
      scale_color_viridis(discrete = TRUE)
  }
  return(p)
}

# Presents SSR
present_ssr <- function(mods, table_theme = "mBlue", footnote = "Model R-squared", visual = TRUE) {
  mods_summ <- mods %>%
    map(~summary(.))
  SSR <<- tibble("Model" = Qsn_Name_Cont,  
                 "SSR" = round(mods_summ %>%
                                 map_dbl(~sum(.$residuals^2)/.$df[2]), digits = 2))
  if(!visual){
    p <- ggtexttable(SSR, theme = ttheme(table_theme), rows = NULL) %>%
      tab_add_footnote(text = footnote, size = 10, face = "italic")}
  else{
    p <- SSR %>%
      mutate(Model = as.factor(Model)) %>%
      ggplot(aes(fct_reorder(Model, SSR, .desc = TRUE), SSR, group = 1)) +
      geom_line(size = 1.5) +
      geom_point(size = 2.5) +
      theme_bw() +
      theme(plot.title = element_text(hjust = 0.5), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),strip.background = element_blank()) +
      ggtitle("Model Residuals") +
      theme(
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        strip.background = element_blank()
      ) + 
      labs(
        x = "Model",
        y = "Sum of Squares Residuals"
      )
  }
  return(p)
}

format_reg_table <- function(model, digs = 2){
  mod <- tidy(model, conf.level = .99, conf.int = TRUE) %>%
    mutate(across(where(is.numeric), ~ round(., digits = digs)))
  mod <- mod %>%
    mutate(estimate = as.character(estimate)) %>%
    mutate(estimate = paste(paste(estimate, ifelse(p.value < 0.05, "*",""), ifelse(p.value < 0.01, "*",""), ifelse(p.value < 0.001, "*",""), sep = ""), paste("[", conf.low, ", ", conf.high, "]", sep = ""), sep = "\n"))
  mod <- mod %>%
    select(term, estimate)
  return(mod)
}

# Presents Coefficients
present_reg_mod <- function(model, title = "", table_theme = "mOrange") {
  p <- ggtexttable(format_reg_table(model), theme = ttheme(table_theme), rows = NULL) %>%
    tab_add_title(text = title, face = "bold", padding = unit(0.1, "line")) %>%
    tab_add_footnote(text = paste("*** p < 0.001; ** p < 0.01;", " * p < 0.05; 99% C.I.", sep = "\n"), size = 10, face = "italic")
  return(p)
}