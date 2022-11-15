### Steps to run the script:

#### Install the requirements
```bash
pip install -r requirements.txt
```
#### Script usage

```bash
(py39) bash-3.2$ python train.py -h
usage: train.py [-h] [--exclude-outliers] [--include-outliers] [-r REMOVE_PERCENT] [-n FILLNA_METHOD] [--normal-process]

optional arguments:
  -h, --help            show this help message and exit
  --exclude-outliers    Pass this variable to exclude outliers in the data
  --include-outliers    Pass this variable to include outliers in the data
  -r REMOVE_PERCENT, --remove-percent REMOVE_PERCENT
                        Pass the number of percentage that you want to remove from the data
  -n FILLNA_METHOD, --fillna-method FILLNA_METHOD
                        Pass the method that you want to use to fill the missing values
  --normal-process      Pass this variable to check the scores in the usual process
```

#### Commands:
```bash
python train.py -h
python train.py --include-outliers --remove-percent 0 --fillna-method None --normal-process 
python train.py --include-outliers --remove-percent 0 --fillna-method None --normal-process
python train.py --exclude-outliers --remove-percent 0 --fillna-method None --normal-process
python train.py --exclude-outliers --remove-percent 0 --fillna-method None --normal-process
python train.py --exclude-outliers --remove-percent 0 --fillna-method None --normal-process
python train.py --include-outliers --remove-percent 0 --fillna-method None --normal-process
python train.py --exclude-outliers --remove-percent 1 --fillna-method bfill
python train.py --exclude-outliers --remove-percent 5 --fillna-method bfill
python train.py --exclude-outliers --remove-percent 10 --fillna-method bfill
python train.py --exclude-outliers --remove-percent 1 --fillna-method mode
python train.py --exclude-outliers --remove-percent 5 --fillna-method mode
python train.py --exclude-outliers --remove-percent 10 --fillna-method mode
python train.py --exclude-outliers --remove-percent 1 --fillna-method ffill
python train.py --exclude-outliers --remove-percent 5 --fillna-method ffill
python train.py --exclude-outliers --remove-percent 10 --fillna-method ffill

```
