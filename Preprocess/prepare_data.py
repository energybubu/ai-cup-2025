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
    # test_ds = load_dataset("acct-fraud-agg-v2", split="test", token=True)
    # alert = load_dataset("fintech-final", split="alert", token=True)
    # non_alert = load_dataset("fintech-final", split="non_alert", token=True)
    test_ds = load_from_disk("acct-fraud-agg-v2")["test"]
    alert = load_from_disk("fintech-final")["alert"]
    non_alert = load_from_disk("fintech-final")["non_alert"]

    # Deterministic seed for reproducible splits
    seed = 42
    non_alert_split = non_alert.train_test_split(test_size=len(alert)*19, seed=seed)
    non_alert_train = non_alert_split["train"]
    non_alert_dev = non_alert_split["test"]

    train_ds = non_alert_train
    dev_ds = concatenate_datasets([alert, non_alert_dev])

    acct_to_split = {}
    acct_to_label = {}
    for example in train_ds:
        acct_to_split[example["acct"]] = 0
        acct_to_label[example["acct"]] = example["label"]
    for example in dev_ds:
        acct_to_split[example["acct"]] = 1
        acct_to_label[example["acct"]] = example["label"]
    for example in test_ds:
        acct_to_split[example["acct"]] = 2
        acct_to_label[example["acct"]] = -1

    df = pd.read_csv("../data/acct_transaction.csv")
    return df, acct_to_split, acct_to_label


def preprocess_data(df, acct_to_split, acct_to_label):
    """"Preprocess the transaction data and save the formatted data and account mapping."""
    currency_mapping = {
        'TWD': 0,
        'USD': 1,
        'AUD': 2,
        'JPY': 3,
        'CNY': 4,
        'GBP': 5,
        'EUR': 6,
        'HKD': 7,
        'THB': 8,
        'SGD': 9,
        'SEK': 10,
        'NZD': 11,
        'CAD': 12,
        'ZAR': 13,
        'CHF': 14,
        'MXN': 15
    }
    channel_mapping = {
        "01": 0,
        "02": 1,
        "03": 2,
        "04": 3,
        "05": 4,
        "06": 5,
        "07": 6,
        "99": 7,
        "UNK": 8
    }
    self_mapping = {
        "Y": 0,
        "N": 1,
        "UNK": 2
    }

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

    # stack all accounts together
    all_accts = pd.concat([
        from_accts.rename(columns={"from_acct": "acct", "from_acct_type": "acct_type"}),
        to_accts.rename(columns={"to_acct": "acct", "to_acct_type": "acct_type"})
    ], ignore_index=True)

    # 2. Drop duplicates while keeping the first type seen
    unique_accts = all_accts.drop_duplicates(subset=["acct"]).reset_index(drop=True)

    # 3. Assign each account an ID (same as len(account_mapping) order)
    unique_accts["acct_id"] = np.arange(len(unique_accts))

    # 4. Build mapping dicts
    acct_to_id = dict(zip(unique_accts["acct"], unique_accts["acct_id"]))
    acct_to_type = dict(zip(unique_accts["acct"], unique_accts["acct_type"]))

    # 5. Map IDs back to df
    df["from_acct"] = df["from_acct"].map(acct_to_id)
    df["to_acct"] = df["to_acct"].map(acct_to_id)

    # (Optional) reconstruct `account_mapping` dict for later use
    account_mapping = {acct_to_id[acct]: (acct_to_split.get(acct, -1), acct_to_label.get(acct, 0), acct) for acct in acct_to_id}

    df["txn_amt_to_twd"] = df.apply(
        lambda row: row["txn_amt"] * twd_per_currency.get(row["currency_type"], 1.0),
        axis=1
    )

    df["timestamp"] = df.apply(convert_to_minutes, axis=1)
    df.to_csv("../data/formatted_transaction.csv")

    with open("../data/account_mapping.json", "w") as f:
        json.dump(account_mapping, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    """Main function to load and preprocess data."""
    
    df, acct_to_split, acct_to_label = load_data()
    preprocess_data(df, acct_to_split, acct_to_label)    
