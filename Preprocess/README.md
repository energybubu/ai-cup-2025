# Preprocess

此資料夾下有2個檔案 `date_split.py`, `prepare_data.py`，用於資料前處理。
首先 `acct_alert.csv`, `acct_predict.csv`, `acct_transaction.csv` 應放在 `data` 此資料夾下，結構如：
```
.
├ data
    ├ acct_alert.csv
    ├ acct_predict.csv
    ├ acct_transaction.csv
├ Preprocess
    ├ prepare_data.ipynb
    ├ prepare_data.py
...
```
執行以下指令：
```
python date_split.py
```
會儲存兩個 `DatasetDict` 在 `Preprocess` 資料夾下，分別為 `acct-fraud-agg-v2` 以及 `fintech-final`。 
```
python prepare_data.py
```
會讀取`acct-fraud-agg-v2` 以及 `fintech-final` 並輸出2個 file `formatted_transaction.csv`, `account_mapping.json` 存放於 `data` 資料夾下。