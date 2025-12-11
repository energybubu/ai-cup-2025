from datasets import load_dataset, concatenate_datasets, load_from_disk
import pandas as pd
from datetime import datetime
import numpy as np
import json


def convert_to_minutes(row):
    """Convert transaction date and time to total minutes."""
    h, m, s = map(int, row["txn_time"].split(":"))
    return row["txn_date"] * 24 * 60 + h * 60 + m + s / 60


def load_data():
    """Load datasets and prepare account split and label mappings."""
    test_ds = load_from_disk("acct-fraud-agg-v2_register")["test"]
    alert = load_from_disk("fintech-final_register")["alert"]
    non_alert = load_from_disk("fintech-final_register")["non_alert"]

    # Deterministic seed for reproducible splits
    seed = 42
    # non_alert_split = non_alert.train_test_split(test_size=len(alert)*19, seed=seed)
    # non_alert_train = non_alert_split["train"]
    # non_alert_dev = non_alert_split["test"]

    # train_ds = non_alert_train
    # dev_ds = concatenate_datasets([alert, non_alert_dev])

    acct_to_split = {}
    acct_to_label = {}
    # for example in train_ds:
    #     acct_to_split[example["acct"]] = 0
    #     acct_to_label[example["acct"]] = example["label"]
    # for example in dev_ds:
    #     acct_to_split[example["acct"]] = 1
    #     acct_to_label[example["acct"]] = example["label"]

    for example in alert:
        acct_to_split[example["acct"]] = 1
        acct_to_label[example["acct"]] = 1
    for example in non_alert:
        acct_to_split[example["acct"]] = 1
        acct_to_label[example["acct"]] = 0
    for example in test_ds:
        acct_to_split[example["acct"]] = 2
        acct_to_label[example["acct"]] = -1

    register_df = pd.read_csv("../data/acct_register.csv")
    # collect all accounts in register file
    for _, row in register_df.iterrows():
        if row["from_acct"] not in acct_to_split:
            acct_to_split[row["from_acct"]] = 1
            acct_to_label[row["from_acct"]] = 0
        if row["to_acct"] not in acct_to_split:
            acct_to_split[row["to_acct"]] = 1
            acct_to_label[row["to_acct"]] = 0

    df = pd.read_csv("../data/acct_transaction.csv")
    return df, register_df, acct_to_split, acct_to_label


def preprocess_data(df, register_df, acct_to_split, acct_to_label):
    """"Preprocess the transaction data and save the formatted data and account mapping."""

    # Exchange rates TWD → other currencies
    rates_to_twd = {
        'TWD': 1,
        'USD': 0.0328,
        'AUD': 0.0497,
        'JPY': 4.8497,
        'CNY': 0.2336,
        'GBP': 0.0244,
        'EUR': 0.0280,
        'HKD': 0.2553,
        'THB': 1.0623,
        'SGD': 0.0332,
        'SEK': 0.2817,
        'NZD': 0.0500,
        'CAD': 0.0332,
        'ZAR': 0.5957,
        'CHF': 0.0300,
        'MXN': 0.5945
    }
    # Store reciprocals (TWD per 1 unit of currency)
    twd_per_currency = {k: 1/v for k, v in rates_to_twd.items()}

    df["from_acct_type"] = df["from_acct_type"] - 1
    df["to_acct_type"] = df["to_acct_type"] - 1

    # 1. Collect all unique accounts and their types
    from_accts = df[["from_acct", "from_acct_type"]]
    to_accts = df[["to_acct", "to_acct_type"]]

    # register_df has no acct_type, fill with nan/default value
    register_df["from_acct_type"] = 1
    register_df["to_acct_type"] = 1
    register_from_accts = register_df[["from_acct", "from_acct_type"]].rename(columns={"from_acct": "acct", "from_acct_type": "acct_type"})
    register_to_accts = register_df[["to_acct", "to_acct_type"]].rename(columns={"to_acct": "acct", "to_acct_type": "acct_type"})

    # stack all accounts together
    all_accts = pd.concat([
        from_accts.rename(columns={"from_acct": "acct", "from_acct_type": "acct_type"}),
        to_accts.rename(columns={"to_acct": "acct", "to_acct_type": "acct_type"}),
        register_from_accts,
        register_to_accts
    ], ignore_index=True)

    # 2. Drop duplicates while keeping the first type seen
    unique_accts = all_accts.drop_duplicates(subset=["acct"]).reset_index(drop=True)

    # 3. Assign each account an ID (same as len(account_mapping) order)
    unique_accts["acct_id"] = np.arange(len(unique_accts))

    # 4. Build mapping dicts
    acct_to_id = dict(zip(unique_accts["acct"], unique_accts["acct_id"]))
    # acct_to_type = dict(zip(unique_accts["acct"], unique_accts["acct_type"]))

    # 5. Map IDs back to df
    df["from_acct"] = df["from_acct"].map(acct_to_id)
    df["to_acct"] = df["to_acct"].map(acct_to_id)

    register_df["from_acct"] = register_df["from_acct"].map(acct_to_id)
    register_df["to_acct"] = register_df["to_acct"].map(acct_to_id)

    # (Optional) reconstruct `account_mapping` dict for later use
    account_mapping = {acct_to_id[acct]: (acct_to_split.get(acct, -1), acct_to_label.get(acct, 0), acct) for acct in acct_to_id}

    df["txn_amt_to_twd"] = df.apply(
        lambda row: row["txn_amt"] * twd_per_currency.get(row["currency_type"], 1.0),
        axis=1
    )

    df["timestamp"] = df.apply(convert_to_minutes, axis=1)
    df.to_csv("../data/formatted_transaction_register.csv")
    register_df.to_csv("../data/formatted_register.csv")

    with open("../data/account_mapping_register.json", "w") as f:
        json.dump(account_mapping, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    """Main function to load and preprocess data."""
    
    df, register_df, acct_to_split, acct_to_label = load_data()
    preprocess_data(df, register_df, acct_to_split, acct_to_label)    
