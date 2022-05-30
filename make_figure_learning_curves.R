# Johannes Hertzler

# libraries
library(tidyverse)

# set plot theme
theme_set(theme_light())


#####################################################################
# helper functions

# rename cols in train_loss, pivot data to long format and mutate cols
loss_to_long <- function(dt) {
        
    colnames_new <- paste0('epoch_', colnames(dt[-1]))
    colnames(dt) <- c('batch', colnames_new)
        
    dt_long <- dt %>%
        pivot_longer(
            cols = starts_with('epoch'),
            names_to = 'epoch',
            values_to = 'loss'
        ) %>%
        mutate(
            epoch = str_split_fixed(epoch, '_', n = 2)[, 2],
            epoch = as.numeric(epoch) + 1
        ) %>%
        arrange(epoch, batch) %>%
        mutate(
            running_batch = 1:nrow(.)
        )
    return(dt_long)
}


#####################################################################
# set paths
read_path = 'code/output/'
write_path = 'code/output/'


#####################################################################
# load data

files <- list.files(read_path, recursive = T, full.names = T)
files

# loss and validation data
(mob_losses <- read_csv(
     paste0(read_path, 'mob_net/mob/mob_net_losses.csv')
 ) %>%
     loss_to_long() %>%
     mutate(network = 'mobileNet')
)

(mob_val <- read_csv(
     paste0(read_path, 'mob_net/mob/mob_net_val.csv')
 ) %>%
     rename(., 'epoch' = 1) %>%
     mutate(
         epoch = epoch + 1,
         network = 'mobileNet'
     )
)

(mob_ca_losses <- read_csv(
     paste0(read_path, 'mob_net/mob_ca/mob_net_ca_losses.csv')
 ) %>%
     loss_to_long() %>%
     mutate( network = 'mobileNet_ca')
)

(mob_ca_val <- read_csv(
     paste0(read_path, 'mob_net/mob_ca/mob_net_ca_val.csv')
 ) %>%
     rename(., 'epoch' = 1) %>%
     mutate(
         epoch = epoch + 1,
         network = 'mobileNet_ca'
     )
)

(eff_losses <- read_csv(
     paste0(read_path, 'eff_net/eff_net_losses.csv')
 ) %>%
     loss_to_long() %>%
     mutate(network = 'efficientNet')
)

(eff_val <- read_csv(
     paste0(read_path, 'eff_net/eff_net_val.csv')
 ) %>%
     rename(., 'epoch' = 1) %>%
     mutate(
         epoch = epoch + 1,
         network = 'efficientNet'
     )
)


#####################################################################
# plot learning curves 


train_losses <- bind_rows(
    mob_losses,
    mob_ca_losses,
    eff_losses
) %>%
    rename(
        train_loss = loss
    )

val <- bind_rows(
    mob_val,
    mob_ca_val,
    eff_val
) %>%
    rename(
        val_loss = mean_loss
    )

train_losses <- train_losses %>%
    mutate(
        join_epoch = ifelse(batch == max(batch), epoch, NA)
    )

train <- left_join(
    train_losses,
    val,
    by = c('network', c('join_epoch' = 'epoch'))
) %>%
    pivot_longer(
        cols = ends_with('loss'),
        names_to = 'loss_type',
        values_to = 'loss'
    ) %>%
    mutate(
        loss_type = recode(
            loss_type,
            'train_loss' = 'train',
            'val_loss' = 'validation'
        ),
        network = recode(
            network,
            'mobileNet_ca' = 'mobileNet class-aware'
        )
    )


# scale breaks on second y axis
breaks_sec_y = c(50, 60, 70, 80) / 3 * 10 / 100


# make plot
p_leanring_curves <- train %>%
    ggplot(
        aes(epoch, accuracy / 3 * 10),
        col = 'blue') +
    geom_smooth(
        method = 'loess',
        se = T,
        alpha = .3
    ) +
    geom_smooth(
        method = 'loess',
        aes(running_batch / nrow(mob_losses) * 15,
            loss,
            col = loss_type),
        alpha = .3
    ) +
    scale_x_continuous(
        sec.axis = sec_axis(~ . * nrow(mob_losses) / 15,
                            name = element_blank()),
    ) +
    scale_y_continuous(
            sec.axis = sec_axis(~ . * 3 / 10 * 100,
                                name = 'validation accuracy / %',
                                breaks = c(50, 60,  70, 70, 80)),
        minor_breaks = c(50, 60, 70, 80) / 3 * 10 / 100,
        breaks = seq(.6, 1.6, .2)
    ) +
    facet_wrap(~network) +
    scale_colour_viridis_d(end = .8) +
    labs(col = 'loss:',
         x = 'epoch',
         y = 'loss') +
     theme(axis.title.y.right = element_text(colour = 'blue'),
           axis.text.y.right = element_text(colour = 'blue'),
           axis.ticks.y.right = element_line(colour = 'blue'),
           panel.grid.minor.y = element_line(colour = 'blue'),
           axis.text.x.top = element_blank(),
           axis.ticks.x.top = element_blank()) +
    theme(legend.position = 'bottom')

p_leanring_curves


# save plot
ggsave(
    paste0(write_path, 'learning_curves.png'),
    plot = p_leanring_curves,
    width = 19,
    height = 11,
    units = 'cm'
)
