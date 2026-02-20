"""
DSA 2026 Entry Assessment - Section 1: Economic Data Analysis
Solutions for Questions E1 to E5

HOW TO USE:
Copy each block below into the corresponding cell in DSA_2026_Entry.ipynb.
Do NOT copy this entire file as one cell.
"""

# =============================================================================
# QUESTION E1 — Load the Dataset
# =============================================================================

import pandas as pd
import numpy as np

df = pd.read_excel('data/uganda-consumer-price-index-trends-2020-2023.xlsx')

print("Shape of dataframe:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))


# =============================================================================
# QUESTION E2 — Reshape from Wide to Long Format
# =============================================================================

df_long = df.melt(
    id_vars=['indicator_code', 'description'],
    var_name='date',
    value_name='value'
)

print("Shape after reshaping:", df_long.shape)
print("\nFirst 10 rows:")
print(df_long.head(10))


# =============================================================================
# QUESTION E3 — Convert Date Column to Datetime
# =============================================================================

df_long['date'] = pd.to_datetime(df_long['date'], format='%b-%y')

print("Date column info:")
print(df_long['date'].head(10))
print("\nDate range:", df_long['date'].min(), "to", df_long['date'].max())


# =============================================================================
# QUESTION E4 — Filter CPI Indicators
# =============================================================================

df_cpi = df_long[df_long['indicator_code'].str.startswith('CPI_')]

df_all_items = df_cpi[df_cpi['indicator_code'].isin(['CPI_16', 'CPI_09'])]
df_core      = df_cpi[df_cpi['indicator_code'].isin(['CPI_CORE_16', 'CPI_CORE_09'])]
df_food      = df_cpi[df_cpi['indicator_code'].isin(['CPI_FOOD_16', 'CPI_FOOD_09'])]
df_efu       = df_cpi[df_cpi['indicator_code'].isin(['CPI_EFU_16', 'CPI_EFU_09'])]

print("All Items CPI shape:", df_all_items.shape)
print("Core CPI shape:", df_core.shape)
print("Food CPI shape:", df_food.shape)
print("EFU CPI shape:", df_efu.shape)


# =============================================================================
# QUESTION E5 — Handle Missing Values
# =============================================================================

missing_before = df_long['value'].isna().sum()
print("Missing values before replacement:", missing_before)

df_long['value'] = df_long.groupby('indicator_code')['value'].transform(
    lambda x: x.fillna(x.median())
)

missing_after = df_long['value'].isna().sum()
print("Missing values after replacement:", missing_after)
