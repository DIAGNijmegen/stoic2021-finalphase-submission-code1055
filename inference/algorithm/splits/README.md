# Splits

## split

The original split. Contains equal ratio of severe cases in training and validation set.

## inf_split

Subset of `split.csv` that contains only mild and severe cases.

## split_2sev

Same as `inf_split.csv` but severe cases are duplicated for rebalancing.

## split_meta

Evenly split (same distribution within training set and validation set), considering probCOVID, probSevere, PatientAge, and PatientSex.

Contains no rebalancing.

## split_meta_sevbalance

Same as `split_meta.csv` but contains a balanced training set (regarding probSevere). The training set contains no uninfected patients at all. The validation set is the same as `split_meta.csv`.

## split_meta8
Same as `split_meta.csv` but with an 8-fold cross-validation.

## split_meta_sevbalance8
Same as `split_meta8.csv` but contains a balanced training set (regarding probSevere). The training set contains no uninfected patients at all. The validation set is the same as `split_meta8.csv`.

## split_sev_cv8
8-fold cross validation without extra test set. Contains only infected patients and is balanced towards 1:1 distribution of severe cases.

## split_sev_cv5
5-fold cross validation without extra test set. Contains only infected patients and is balanced towards 1:1 distribution of severe cases.

## sevbal_cv5
5-fold cross validation without extra test set. Contains patients without COVID infection. Infected patients are balanced and have a 1:1 severe:mild ratio.

## sevbal_cv5_val=0
Almost the same as `sevbal_cv5.csv` but fold 0 serves as the validation set. The validation set contains no duplicates.
