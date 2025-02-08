# Data Analysis

## Missing data

Many features related to RGB mean and standard deviation per date (img_red_mean_date1, img_blue_std_date5, etc.) have 1,954 missing values.

Date and Change Status Columns (date0, change_status_date0, etc.) also contain missing values.

## Categorical variables

urban_type and geography_type contain multiple labels per entry (e.g., 'Sparse Urban,Industrial').

change_type is straightforward, but dates (date0, date1, ...) should be converted into datetime format.

## Geometry Feature Extraction

Extracted area and perimeter—good step, but additional geometry-based features might be useful.