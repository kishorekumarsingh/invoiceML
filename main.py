# Following this medium article for EDA
# https://towardsdatascience.com/exploratory-data-analysis-in-python-c9a77dfa39ce

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', False)
sns.set(color_codes=True)

raw_df = pd.read_csv("H2HBABBA2687.csv")

# Drop all the unnecessary columns
col_dropped_df = raw_df.drop([
    "name_customer",
    "doc_id",
    "area_business",
    "invoice_id",
    "buisness_year",
    "posting_id",
], axis=1
)

# Rename columns to make more readable and meaningful
renamed_df = col_dropped_df.rename(columns={
    "due_in_date": "due_date",
    "invoice_currency": "currency",
    "total_open_amount": "amount",
    "document type": "document_type",
    "isOpen": "is_open"
})

# Find the number of duplicated rows
duplicated_df = renamed_df[renamed_df.duplicated()]

# Remove duplicated rows
unique_df = renamed_df.drop_duplicates()

# Remove all the rows for open invoices, this data cannot be used either for training, validation or testing the model.
unique_df = unique_df[unique_df.is_open == 0].drop(["is_open"], axis=1)


# Function to convert columns to date
def convertToDate(df, col):
    for k, v in col.items():
        df[k] = pd.to_datetime(unique_df[k], format=v)
    return df


unique_df['clear_date'] = unique_df['clear_date'].str.split().str[0]
unique_df = convertToDate(unique_df, {
    "document_create_date": "%Y%m%d",
    "document_create_date.1": "%Y%m%d",
    "due_date": "%Y%m%d",
    "baseline_create_date": "%Y%m%d",
    "clear_date": "%d-%m-%Y",
    "posting_date": "%d-%m-%Y",
    }
)
unique_df = unique_df[unique_df["document_type"].str.contains("RV")].drop(["document_type"], axis=1)

unique_df.to_csv("final_dataset.csv")

# Add a column for difference in due_date and clear_date
unique_df["diff"] = (unique_df["clear_date"] - unique_df["due_date"]).dt.days

# Remove outliers
Q1 = unique_df["diff"].quantile(0.01)
Q3 = unique_df["diff"].quantile(0.99)
unique_df = unique_df[(unique_df["diff"] > Q1) & (unique_df["diff"] < Q3)]


# fig, axes = plt.subplots(1, 2, figsize=(18, 6))
#
# ax = axes[0]
# df = unique_df["currency"].value_counts()
# ax.pie(df, autopct='%1.1f%%', labels=df.index, startangle=30)
# ax.legend()
#
# ax = axes[1]
# df = unique_df["business_code"].value_counts(normalize=True)
# ax.pie(df, autopct='%1.1f%%', labels=df.index, startangle=30)
# ax.legend()
#
# plt.show()
#
# df = unique_df[{"currency", "diff"}]
# df = df.groupby(['currency'])['diff'].mean()
# ax = df.plot(kind='bar', figsize=(20, 10), color="indigo", fontsize=13)
# ax.set_alpha(0.8)
# ax.set_title("Days-past-due VS currency", fontsize=22)
# ax.set_xlabel("currency", fontsize=15)
# ax.set_ylabel("Number of days past due date", fontsize=15)
# plt.show()
#
# df = unique_df[{"business_code", "diff"}]
# df = df.groupby(['business_code'])['diff'].mean()
# ax = df.plot(kind='bar', figsize=(20, 10), color="indigo", fontsize=13)
# ax.set_alpha(0.8)
# ax.set_title("Days-past-due VS business_code", fontsize=22)
# ax.set_xlabel("business_code", fontsize=15)
# ax.set_ylabel("Number of days past due date", fontsize=15)
# plt.show()
#
#
# df = unique_df[{"cust_payment_terms", "diff"}]
# df = df.groupby(['cust_payment_terms'])['diff'].mean()
# ax = df.plot(kind='bar', figsize=(20, 10), color="indigo", fontsize=13)
# ax.set_alpha(0.8)
# ax.set_title("Days-past-due VS cust_payment_terms", fontsize=22)
# ax.set_xlabel("cust_payment_terms", fontsize=15)
# ax.set_ylabel("Number of days past due date", fontsize=15)
# plt.show()
#
#


