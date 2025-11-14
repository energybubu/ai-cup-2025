import numpy as np
import pandas as pd
import polars as pl
from datasets import Dataset, DatasetDict
from tqdm import tqdm


# Check ratio of existing transaction after cutoff_date among non-alerted accounts
def check_non_alerted_txn_ratio(aggregated_data, alerted_accts, cutoff_date):
    """Checks the ratio of non-alerted accounts that have transactions after the cutoff date."""

    total_non_alerted = 0
    non_alerted_with_txn_after_cutoff = 0

    # derive non-alerted accounts from provided aggregated_data and alerted_accts
    non_alerted_accts = set(aggregated_data.keys()) - set(alerted_accts)

    for acct in non_alerted_accts:
        total_non_alerted += 1
        # Check if there's any transaction after the cutoff date
        if any(
            txn["txn_date"] > cutoff_date
            for txn in aggregated_data[acct]["transactions"]
        ):
            non_alerted_with_txn_after_cutoff += 1

    if total_non_alerted == 0:
        return 0.0

    return non_alerted_with_txn_after_cutoff / total_non_alerted


# 1. Aggregate transactions for each account.
def aggregate_transactions_to_list(df):
    """Aggregates transactions for each account into a list of transactions."""

    accounts_data = {}

    for row in df.iter_rows(named=True):
        # Transform txn_time to int, representing minutes.
        row["txn_time"] = int(row["txn_time"].split(":")[0]) * 60 + int(
            row["txn_time"].split(":")[1]
        )
        # Only save accounts of type '01' (玉山帳號)
        # Transaction from the sender's perspective
        if int(row["from_acct_type"]) == 1:
            from_acct = row["from_acct"]
            if from_acct not in accounts_data:
                accounts_data[from_acct] = {"transactions": []}

            outgoing_txn = row.copy()
            outgoing_txn["txn_type"] = 0  # "outgoing"
            outgoing_txn["counterparty_type"] = (
                1 if int(row["to_acct_type"]) == 1 else 0
            )
            accounts_data[from_acct]["transactions"].append(outgoing_txn)

        # Transaction from the receiver's perspective
        if int(row["to_acct_type"]) == 1:
            to_acct = row["to_acct"]

            if to_acct not in accounts_data:
                accounts_data[to_acct] = {"transactions": []}

            incoming_txn = row.copy()
            incoming_txn["txn_type"] = 1  # "incoming"
            incoming_txn["counterparty_type"] = (
                1 if int(row["from_acct_type"]) == 1 else 0
            )
            accounts_data[to_acct]["transactions"].append(incoming_txn)

    return accounts_data


def load_and_process_data():
    """Load and sort transaction, process alert, and test account data."""

    # Load transaction data
    print("Loading transaction data...")
    txns = pl.read_csv("../data/acct_transaction.csv")
    aggregated_data = aggregate_transactions_to_list(txns)

    # Sort transactions for each account by txn_date and txn_time
    for acct in tqdm(aggregated_data, desc="Sorting transactions by date"):
        aggregated_data[acct]["transactions"].sort(
            key=lambda x: (
                x["txn_date"],
                x["txn_time"],
            )
        )

    # Load alert data and test account
    print("Loading alert data...")
    alerts = pd.read_csv("../data/acct_alert.csv")
    alerts = alerts.sort_values("event_date").reset_index(drop=True)
    test_account = set(pd.read_csv("../data/acct_predict.csv")["acct"])
    print("length of test_account:", len(test_account))

    # filter out test account from aggregated_data
    test_data = [
        {
            "acct": acct,
            "transactions": aggregated_data[acct]["transactions"],
            "label": 0,
        }
        for acct in tqdm(test_account, desc="Processing test accounts")
        if acct in aggregated_data
    ]
    print("length of test_data:", len(test_data))
    assert len(test_account) == len(test_data)

    aggregated_data = {
        acct: data for acct, data in aggregated_data.items() if acct not in test_account
    }
    return aggregated_data, alerts, test_account, test_data


def store_alert_and_non_alert_data(aggregated_data, alerts):
    """Store full alert and non-alert data to disk."""

    # Get all accounts from aggregated data
    print("Get all accounts from aggregated data")
    all_accts = set(aggregated_data.keys())
    non_alerted_accts = all_accts - set(alerts["acct"])

    alert_data = [
        {
            "acct": acct,
            "transactions": aggregated_data[acct]["transactions"],
            "label": 1,
        }
        for acct in tqdm(
            alerts["acct"], desc="Processing alerted accounts for full alert_data"
        )
    ]
    non_alert_data = [
        {
            "acct": acct,
            "transactions": aggregated_data[acct]["transactions"],
            "label": 0,
        }
        for acct in tqdm(
            non_alerted_accts, desc="Processing non-alerted accounts for full alert_data"
        )
    ]
    print(len(alert_data), len(non_alert_data))

    alert_dataset = Dataset.from_list(alert_data)
    non_alert_dataset = Dataset.from_list(non_alert_data)

    dataset_dict = DatasetDict(
        {
            "alert": alert_dataset,
            "non_alert": non_alert_dataset,
        }
    )
    dataset_dict.save_to_disk("fintech-final")
    return all_accts, non_alerted_accts


def split_and_store_data(
    aggregated_data,
    alerts,
    non_alerted_accts,
):
    """Split data into train/dev sets and store to disk."""

    print("Splitting data into train/dev sets...")
    # 2. Randomly sample alerted accounts into train/dev sets
    # We'll randomly sample 80% of alerted events for training and use the remaining 20% for dev.
    # Use a fixed random_state for reproducibility.
    train_alerts = alerts.sample(frac=0.8, random_state=42)
    dev_alerts = alerts.drop(train_alerts.index)

    # up-sample train set to 3x
    train_alerts = pd.concat([train_alerts] * 3, ignore_index=True)

    # Sort and reset indices for downstream consistency
    train_alerts = train_alerts.sort_values("event_date").reset_index(drop=True)
    dev_alerts = dev_alerts.sort_values("event_date").reset_index(drop=True)

    print("Number of alerted accounts total:", len(alerts))
    print("Number of alerted accounts in training sample:", len(train_alerts))
    print("Number of alerted accounts in dev sample:", len(dev_alerts))

    # # Determine the cutoff date for non-alerted accounts
    # cutoff_date = train_alerts["event_date"].max()

    # Get the sets of alerted accounts
    train_alerted_accts = set(train_alerts["acct"])
    dev_alerted_accts = set(dev_alerts["acct"])
    print("Number of alerted accounts in training set:", len(train_alerted_accts))
    print("Number of alerted accounts in dev set:", len(dev_alerted_accts))

    # Prepare training and dev data lists
    train_data = []
    dev_data = []

    # Get Train/Dev non-alerted accounts
    # Split into 8/2 and then random sample 19x of the alerted accounts from it.
    split_idx = int(0.8 * len(non_alerted_accts))
    non_alerted_accts_list = list(non_alerted_accts)
    rng.shuffle(non_alerted_accts_list)
    train_non_alerted_accts = non_alerted_accts_list[:split_idx]
    sampled_train_non_alerted_accts = rng.choice(
        train_non_alerted_accts, size=19 * len(train_alerted_accts), replace=False
    )
    dev_non_alerted_accts = non_alerted_accts_list[split_idx:]
    sampled_dev_non_alerted_accts = rng.choice(
        dev_non_alerted_accts, size=19 * len(dev_alerted_accts), replace=False
    )
    assert (
        set(sampled_train_non_alerted_accts) & set(sampled_dev_non_alerted_accts) == set()
    )
    assert len(sampled_train_non_alerted_accts) == 19 * len(train_alerted_accts)
    assert len(sampled_dev_non_alerted_accts) == 19 * len(dev_alerted_accts)
    print("Train non-alerted accounts:", len(sampled_train_non_alerted_accts))
    print("Dev non-alerted accounts:", len(sampled_dev_non_alerted_accts))

    assert len(train_alerted_accts & dev_alerted_accts) == 0
    assert len(dev_alerted_accts & set(dev_non_alerted_accts)) == 0
    assert len(test_account & train_alerted_accts) == 0
    assert len(test_account & dev_alerted_accts) == 0
    assert len(test_account & set(dev_non_alerted_accts)) == 0
    assert len(test_account & set(train_non_alerted_accts)) == 0


    # 3. Process training data
    # Alerted accounts for training
    train_data.extend(
        [
            {
                "acct": acct,
                "transactions": aggregated_data[acct]["transactions"],
                "label": 1,
            }
            for acct in tqdm(
                train_alerted_accts, desc="Processing alerted accounts for training"
            )
        ]
    )

    # Non-alerted accounts for training (transactions before or on cutoff_date)
    train_data.extend(
        [
            {
                "acct": acct,
                "transactions": aggregated_data[acct]["transactions"],
                "label": 0,
            }
            for acct in tqdm(
                sampled_train_non_alerted_accts,
                desc="Processing non-alerted accounts for training",
            )
        ]
    )

    # 4. Process dev data
    # Alerted accounts for developing
    dev_data.extend(
        [
            {
                "acct": acct,
                "transactions": aggregated_data[acct]["transactions"],
                "label": 1,
            }
            for acct in tqdm(
                dev_alerted_accts, desc="Processing alerted accounts for developing"
            )
        ]
    )

    # Non-alerted accounts for developing
    dev_data.extend(
        [
            {
                "acct": acct,
                "transactions": aggregated_data[acct]["transactions"],
                "label": 0,
            }
            for acct in tqdm(
                sampled_dev_non_alerted_accts,
                desc="Processing non-alerted accounts for developing",
            )
        ]
    )

    # # 5. Upload the resulting train/dev splits onto Hugging Face datasets
    # print("Uploading to Hugging Face datasets...")
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset,
        }
    )

    # Save to local folder
    dataset_dict.save_to_disk("acct-fraud-agg-v2")


if __name__ == "__main__":
    """Main function to load and preprocess data."""

    rng = np.random.default_rng(42)

    aggregated_data, alerts, test_account, test_data = load_and_process_data()

    # Store full alert and non-alert data
    all_accts, non_alerted_accts = store_alert_and_non_alert_data(aggregated_data, alerts)

    split_and_store_data(aggregated_data, alerts, non_alerted_accts)