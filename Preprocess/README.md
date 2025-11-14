# Preprocess

This folder contains two files, `date_split.py` and `prepare_data.py`, which are used for data preprocessing.

First, place the files `acct_alert.csv`, `acct_predict.csv`, and `acct_transaction.csv` inside the `data` directory.

## data_split.py
Split the accounts and transactions into 2 categories: **alert** and **non-alert**. \
Also produce **train**, **dev** and **test** split.

## prepare_data.py
Convert **accounts** and **transactions** into **nodes** and **edges** on a graph, respectively. \
Define the features for both nodes and edges.

## Execution

Run the following command:
```
python date_split.py
```

This will generate two DatasetDict objects in the Preprocess directory, named acct-fraud-agg-v2 and fintech-final.

Next, run:
```
python prepare_data.py
```

This will load `acct-fraud-agg-v2` and `fintech-final` and output two files, `formatted_transaction.csv` and `account_mapping.json`, saved in the data directory.