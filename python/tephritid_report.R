## ----setup, include = FALSE------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = FALSE)


## ----message = FALSE, warning = FALSE--------------------------------------------------------------------------------------------
# library(reticulate)
if(!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, pander, patchwork, terra, viridis, reticulate)

source_python("C:/Users/HarmerA/OneDrive - MWLR/Documents/data/tephritidML/python/train_test_functions.py")
py = py_run_file("C:/Users/HarmerA/OneDrive - MWLR/Documents/data/tephritidML/python/exp/act_maps2.py")

dat1 = read_csv("C:/Users/HarmerA/OneDrive - MWLR/Documents/data/tephritidML/logs/tephritid_species_v3_1_Xception_200_epoch_transfer_earlyStop_log.csv") %>% 
  pivot_longer(!epoch, names_to = "stat", values_to = "value")
dat2 = read_csv("C:/Users/HarmerA/OneDrive - MWLR/Documents/data/tephritidML/logs/tephritid_species_v3_2_Xception_200_epoch_transfer_earlyStop_log.csv") %>% 
  pivot_longer(!epoch, names_to = "stat", values_to = "value")
dat3 = read_csv("C:/Users/HarmerA/OneDrive - MWLR/Documents/data/tephritidML/logs/tephritid_species_v3_3_Xception_200_epoch_transfer_earlyStop_log.csv") %>% 
  pivot_longer(!epoch, names_to = "stat", values_to = "value")


## ----warning = FALSE, fig.width = 12, fig.cap = "Fig. 1: Accuracy and loss of the three cross-validated Xception models after 30 epochs.", fig.align="center"----
acc_plot = dat1 %>% 
  filter(stat == "accuracy" | stat == "val_accuracy") %>% 
  ggplot(aes(x = epoch, y = value, colour = stat)) +
    geom_line(linewidth = 1) +
    geom_line(data = subset(dat2, stat == "accuracy" | stat == "val_accuracy"), aes(x = epoch, y = value, colour = stat), linewidth = 1) +
    geom_line(data = subset(dat3, stat == "accuracy" | stat == "val_accuracy"), aes(x = epoch, y = value, colour = stat), linewidth = 1) +
    scale_colour_manual(labels = c("Training accuracy", "Validation accuracy"), values = c("#70ad47", "#4472c4")) +
    ylim(c(0,1)) +
    xlab("\nEpoch") +
    ylab("Accuracy\n") +
    theme_classic(base_size = 16) +
    theme(legend.title = element_blank(), legend.position = c(0.75, 0.25))

loss_plot = dat1 %>% 
  filter(stat == "loss" | stat == "val_loss") %>% 
  ggplot(aes(x = epoch, y = value, colour = stat)) +
    geom_line(linewidth = 1) +
    geom_line(data = subset(dat2, stat == "loss" | stat == "val_loss"), aes(x = epoch, y = value, colour = stat), linewidth = 1) +
    geom_line(data = subset(dat3, stat == "loss" | stat == "val_loss"), aes(x = epoch, y = value, colour = stat), linewidth = 1) +
    scale_colour_manual(labels = c("Training loss", "Validation loss"), values = c("#70ad47", "#4472c4")) +
    xlab("\nEpoch") +
    ylab("Loss\n") +
    theme_classic(base_size = 16) +
    theme(legend.title = element_blank(), legend.position = c(0.75, 0.75))

acc_plot|loss_plot


## from sklearn.metrics import classification_report

## from pandas import DataFrame

## 

## DATASET_PATH = 'C:/Users/harmera/OneDrive - MWLR/Documents/data/tephritidML/'

## model_dir = DATASET_PATH + 'models/'

## model_file = model_dir + 'tephritid_species_v3_1_Xception_transfer.h5'

## model_name = 'Xception'

## labels_path = DATASET_PATH + 'tephritid_annotation.csv'

## images_path = DATASET_PATH + 'img_fold/1/'

## test_data_dir = images_path + 'val/'

## 

## answers = test_model(model_file, labels_path, test_data_dir, model_name)

## 

## predicted = [tup[1] for tup in answers[1]]

## y_test = [tup[0] for tup in answers[1]]

## 

## report = classification_report(y_test, predicted, output_dict = True)

## report = DataFrame(report).transpose()


## --------------------------------------------------------------------------------------------------------------------------------
report = py$report
report[30,] = c(NA, NA, 0.6853933, 89)
report %>% 
  mutate_if(is.numeric, round, digits = 2) %>% 
  pander()


## ----warning = FALSE, message = FALSE, fig.width = 12----------------------------------------------------------------------------
idx = which.max(py$preds)
labels = as.vector(unlist(read_csv("C:/Users/HarmerA/OneDrive - MWLR/Documents/data/tephritidML/tephritid_annotation.csv", col_names = FALSE)))
name = labels[idx]
p = paste0(round(max(py$preds) * 100, digits = 1), "%")

flattened = (py$img[,,1]*0.3)+(py$img[,,2]*0.59)+(py$img[,,3]*0.11)

plot(rast(flattened), col = gray.colors(256), asp = 1, legend = FALSE, axes = FALSE)
plot(rast(py$heatmap[,,2]), legend = TRUE, axes = FALSE, col = viridis(256), alpha = 0.4, add = TRUE)
text(x = 1, y = 285, labels = name, pos = 4, cex = 1)
text(x = 1, y = 270, labels = p, pos = 4, cex = 1)

