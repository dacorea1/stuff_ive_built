#!/usr/bin/env Rscript

#
# Example random forest written in R
#

####
##  Global System Vars
####
model_version <- Sys.Date()
run_date_time <- Sys.time()
author <- 'David Corea'

####
##   Modeling Params
##   Predetermined from Cross Validation
####
num_trees_to_grow <- 40
tree_depth <- 158
num_predictors_to_use_on_any_tree <- 5

## 
# Scrappy Documentation Until We Make This Better
# 
#
#
# Model Training and Output Flow:
#  0. Load appropriate libraries & args into script environment
#  1. Validate that a correctly formatted data file exists on which to 
#     train a new model
#  2a. Read data in and, 
#  2b. Process data appropriately for model fit
#  3. Validate an existing regressor model exists and load model from memory
#  4. Make predictions against new data
#  5. Write predictions to s3
#
##

setwd("/datasets/rev")
# prevents floating point numbers from being shown in Scientific Notation
options(scipen = 99999999) 

###
## 0. Load appropriate libraries & args into script environment
###
library(optparse)
suppressMessages(library(randomForest))


# Parse args passed to script
option_list = list(
  make_option(
    c("-f", "--filepath"), type = 'character',
    default = '/rev_raw.csv', 
    help = "The filepath of the resulting dataset from monthly_org_rev.sql",
    metavar = 'character'
  )
)

opt_parser = OptionParser(option_list = option_list)
opts = parse_args(opt_parser)

###
##  1. Validate that a correctly formatted data file exists on which to 
##     train a new model
###

## Check and assert valid file
# Does file exist?
minimum_reqd_column_names <- c('user_id', 'charge_month', 'revenue_type', 'revenue_amount', 
                              'avg_amount_paid_past_two_months',
                              'revenue_frequency', 
                              ... # columns obfuscated
                              )
if(is.null(opts$filepath)){
  # Case when nothing passed to filepath arg
  warning("Please define a valid filepath with the flag: --filepath")
  stop("Script execution halting.  No valid file or s3 file path supplied.")
}else if(!file.exists(opts$filepath)){
  # Is it a valid file path?
  # Case when file doesn't exist in defined path
  warning("Double check that the filepath you've entered is a valid file path")
  warning(paste("Unable to find file at", opts$filepath, sep="\n"))
  stop("Script execution halting.  No valid file or s3 file path supplied.")
}else{
  # Does file contain correct columns
  print(paste("File found at", opts$filepath, sep="\n"))
  ###
  ## 2a. Read data in
  ###
  # [Ch]urn [D]ataset
  rev_raw <- read.csv(file = opts$filepath, stringsAsFactors = F, strip.white = T)
  are_all_important_columns_present <- stringr::str_to_lower(minimum_reqd_column_names) %in% stringr::str_to_lower(names(rev_raw))
  
  # Are fields required for model present and named appropriately?
  if(FALSE %in% are_all_important_columns_present){
    warning("Valid file found with missing fields. At minimum, the following fields (named as they appear) must be present")
    print(minimum_reqd_column_names)
    stop("Ensure fields are named appropriately before invoking this model.")
  }else{
    print("All relevant fields present in CSV")
    print("Proceeding to model development")
  } 
}


###
##  2b. Process data appropriately for model development
###

## Utility functions
bucket_long_tail_industrys_into_other <- function(ch = adc){
  ch.agg <- data.frame(prop.table(table(ch$industry)))
  names(ch.agg) <- c("industry", "freq")
  ch.agg$industry <- as.character(ch.agg$industry)
  ch.agg <- ch.agg[order(ch.agg$freq, decreasing = T),]
  rownames(ch.agg) <- seq(1:nrow(ch.agg))  
  ch.agg$industry <- ifelse(rownames(ch.agg) <= 30, ch.agg$industry, 'other')
  ch.merge <- merge(x = ch, y = ch.agg, by = "industry", all.x = TRUE)
  # Explicitly removing the unnecessary frequency column before returning to 
  # prod data frame
  ch.merge <- ch.merge[, c('user_id', 'charge_month', 'revenue_type', 'revenue_amount', 
                              'avg_amount_paid_past_two_months',
                              'revenue_frequency', 
                              ...)
                       ]
  
  ch <- ch.merge
  return(ch)
}
bucket_long_tail_mktg_channels_into_other <- function(ch = adc){
  ch$marketing_channel_lower <- stringr::str_to_lower(ch$marketing_channel)
  ch.agg <- data.frame(prop.table(table(ch$marketing_channel_lower)))
  names(ch.agg) <- c("channel", "freq")
  ch.agg$channel <- as.character(ch.agg$channel)
  ch.agg$perc <- ch.agg$freq * 100
  ch.agg <- ch.agg[order(ch.agg$perc, decreasing = T),]
  rownames(ch.agg) <- seq(1:nrow(ch.agg))
  ch.agg$marketing_channel_lower <- ifelse(rownames(ch.agg) <= 30, ch.agg$channel, 'other')
  names(ch.agg) <- c('marketing_channel_lower', 'freq', 'perc', 'channel')
  ch.merge <- merge(x = ch, y = ch.agg, 
                    by ='marketing_channel_lower',
                    all.x = TRUE
                    )

  ch.merge <- ch.merge[, c('user_id', 'charge_month', 'revenue_type', 'revenue_amount', 
                              'avg_amount_paid_past_two_months',
                              'revenue_frequency', 
                              ...)
                      ]
  
  ch <- ch.merge
  return(ch)
}

process_data <- function(ad = rev_raw){
  names(ad) <- stringr::str_to_lower(names(rev_raw))
  adc <- ad[complete.cases(ad),] 
  adc <- adc[ nchar(adc$marketing_channel) > 0 ,]
  adc <- adc[ !(grepl('[T,t]est|[T,t]itle', adc$industry)),]
  adc <- adc[adc$enrollment_age_days > 0 ,]
  
  # Customers with at least 3 digits are newer
  adc <- adc[ nchar(adc$user_id) >= 3,]

  adc <- bucket_long_tail_industrys_into_other(ch = adc)
  adc <- bucket_long_tail_mktg_channels_into_other(ch = adc)

  adc$charge_month <- as.Date(as.character(adc$charge_month), format = '%Y-%m-%d')
  adc$org_size_cat <- as.factor(as.character(adc$org_size_cat))
  adc$industry <- as.factor(as.character(adc$industry))
  adc$channel <- as.factor(as.character(adc$channel))  
  adc$revenue_type <- as.factor(as.character(adc$revenue_type))
  adc$revenue_frequency <- as.factor(as.character(adc$revenue_frequency))
  adc$plan_tier <- as.factor(as.character(adc$plan_tier))
  adc$plan_type <-as.factor(as.character(adc$plan_type))
  
  Customers_with_1.9_cc_rate <- unique(adc[adc$revenue_amount < 0, 'user_id'])
  adc <- adc[ !(adc$user_id %in% Customers_with_1.9_cc_rate),]


  # [A]lmost [M]odeling ready dataset
  adc_am <- adc[, c('charge_month', 'revenue_type', 'avg_amount_paid_past_two_months', 'revenue_frequency', 
                    'num_jobs_created',
                    ...)]
  adc_am$charge_month <- as.character(adc_am$charge_month)
  adc_am$enrollment_age_days <- as.character(adc_am$enrollment_age_days)

  adc_agg <- aggregate(adc_am$revenue_amount,
                       by = list(month = adc_am$charge_month,
                                 revenue_type = adc_am$revenue_type,
                                 avg_amount_paid_past_two_months = adc_am$avg_amount_paid_past_two_months,
                        ...
                       ),
                       FUN = mean)

  names(adc_agg) <- c('month',
                      'revenue_type',
                      ...
  )
  

  # rev Dataset - Model Dev Ready 
  adm <- adc_agg
  adm$month <- as.Date(as.character(adm$month), format = "%Y-%m-%d")
  adm$enrollment_age_days <- as.numeric(adm$enrollment_age_days)

  current_month <- lubridate::floor_date(Sys.Date(), unit = 'month')
  # Opting to not use incomplete data from current month
  adm <- adm[adm$month != current_month, ]

  ## Not going to try to predict Customers in the top 5-10% of monthly revenue
  p95_total_revenue_cutoff <- as.numeric(quantile(adm$avg_revenue, probs = .95))
  adm <- adm[adm$avg_revenue < p95_total_revenue_cutoff, ]

  return(adm)
} # end process_data

adm <- process_data(ad = rev_raw)


###
# 3. Validate an existing regressor model exists and load model from memory
###

## Utility functions
generate_mse_against_test_data <- function(predicted_output, test_df){
  test_df$predicted_rev <- predicted_output
  test_df$error <- abs(test_df$predicted_rev - test_df$avg_revenue)
  print("Error Frequency Distribution Summary")
  print(summary(test_df$error))
  print("Error Cumulative Distribution Summary")
  print(quantile(test_df$error, probs = seq(0,1,.05)))
  
 # write.csv(test_df, file = "./datasets/test_set_evaluation.csv", row.names = F, na = "NULL")
  return(summary(test_df$error))
}

plot_feature_importance <- function(tree_model_importance_matrix){
  tmi <- data.frame(tree_model_importance_matrix)
  names(tmi) <- c('perc_inc_mse', 'inc_node_purity')
  tmi <- tmi[order(tmi$perc_inc_mse, decreasing = F),]
  par(mar = c(4,12,4,2))
  bpi <- barplot( (as.numeric(tmi[,1]))^(1/2),
                  names.arg = rownames(tmi),
                  col = '#2196F3',
                  main = "Most Predictive Features in Current Regressor",
                  xlab = "% Increase in Error w/ Feature Excl",
                  horiz = T,
                  las = 2
  )
  par(mar = c(4,4,4,4))
}

# Train/Test Split
set.seed(128901)
tr_i <- sample(1:nrow(adm), (nrow(adm)/3)*2 )
atr <- adm[tr_i,]
at <- adm[-tr_i,]
rm(tr_i)


rf.asp.fit1 <- randomForest(
  avg_revenue ~ revenue_type + 
    revenue_frequency + avg_amount_paid_past_two_months +
    ...        
  data = atr,
  mtry = num_predictors_to_use_on_any_tree,
  importance = T,
  replace = T,
  nodesize = tree_depth,
  ntree = num_trees_to_grow,
  do.trace = 2
) 

model_filepath <- "/asp_randomForest_current.RData"
save(rf.asp.fit1, file = model_filepath)

## Get model into s3
s3file_path <- 's3://...  /saved_models/'
s3_system_command <- paste('aws s3 cp', model_filepath, s3file_path, sep = " ")
system(s3_system_command)
###
# 4. Make predictions against new data
###

rf.pred <- predict(rf.rev.fit1, adm[, c('revenue_type',
                                        'revenue_frequency',
                                        'avg_amount_paid_past_two_months',
                                        ...
                                      )
                                  ],
                                  type = 'response'
                  )

adm$rev_pred <- rf.pred

adm$model_version <- model_version
adm$model_run_date <- run_date_time


###
# Need to add a normalized months_since_enrollment_column
###

library(lubridate)

gen_rev_month_column <- function(adm){
  determine_rev_month <- function(rev_month, enrollment_age){
    am <- as.Date(as.character(rev_month)) 
    enrollment_month <- am - days(enrollment_age)
    months_since_enrollment <- abs(interval(ymd(am), ymd(enrollment_month)) %/% months(1))
    return(months_since_enrollment)  
  }
  
  
  rev_month <- c()
  for(i in seq(1:nrow(adm))){
    # Really useful when testing.  You'll want this turned on
    print(paste("Row", i, sep = " "))
    revm <- determine_rev_month(adm$month[i], adm$enrollment_age_days[i])
    rev_month[ length(rev_month) + 1 ] <- revm
  }
  
  adm$rev_month <- rev_month 
  return(adm)
}

adm <- gen_rev_month_column(adm)

adm <- adm[, c('rev_month',
                'revenue_type',
               ...
               )
           ]



###
# 5. Write data to s3
###

temp_filename <- paste('rev_predictions', '.csv', sep = "")

s3_filepath <- paste(
  "s3://.../model_output/rev/",
  temp_filename,
  sep = ""
)

local_filepath <- paste(
  "/datasets/rev/",
  temp_filename,
  sep = ""
)

write.csv(adm, 
          file = local_filepath,
          row.names = F, quote = F
         )
s3_write_command <- paste('aws s3 cp', 
                          as.character(local_filepath), 
                          as.character(s3_filepath),                        
                          sep = " ")
system(s3_write_command)











