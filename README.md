# MiM-StocR: Momentum-integrated Multi-task Stock Recommendation with Converge-based Optimization

### Environment
Create a Python 3.8 environment using [requirements.txt](requirements.txt)

Training Data will be shared by Google Drive link after double-blind review.

### Reproduce the result

change [backbone] to one from: [LSTM, GATs, HIST]
```
# For CSI300 dataset
python exp/mtl_training.py --method our_method --device cuda:0 --outdir [target_location] --mtm_source_path --model_name [backbone] --mtm_column mtm0604
# For CSI100 dataset
python exp/mtl_training.py --method our_method --device cuda:0 --outdir [target_location] --mtm_source_path ./data/csi100_mtm.pkl --model_name [backbone] --mtm_column mtm0604
```

### MTL Baselines
change [baseline] to one from: [cagrad, dbmtl, uniw]
```
# For CSI300 dataset
python exp/mtl_training.py --method [baseline] --device cuda:0 --outdir [target_location] --mtm_source_path --model_name [backbone] 
# For CSI100 dataset
python exp/mtl_training.py --method [baseline] --device cuda:0 --outdir [target_location] --mtm_source_path ./data/csi100_mtm.pkl --model_name [backbone]
```

### Single task learning
```
# For CSI300 dataset
python exp/regression_training.py --model_name [backbone] --outdir [target_location] --repeat 3 --device cuda:0
# For CSI100 dataset
python exp/regression_training.py --model_name [backbone] --outdir [target_location] --repeat 3 --device cuda:1 --mtm_source_path ./data/csi100_mtm.pkl
```

To use cross-entropy or pair-wise loss function, add ```--loss_type cross-entropy``` or  ```--loss_type pair-wise```

The averaged result will be stored in the log file from result folders.

Reproduce the Qlib backtest:

For MTL baselines (cagrad, our method, dbmtl and equal weight), 
change the ```model_path``` in ```python prediction_mto.py```, 
run ```python prediction_mto.py``` and prediction file will be saved in ```pkl_path```.

For single task learning,
change the ```model_path``` in ```python prediction.py```, 
run ```python prediction.py``` and prediction file will be saved in ```pkl_path```.


Change the pickle file in ```backtest.py``` and run for bakctest and return analysis.