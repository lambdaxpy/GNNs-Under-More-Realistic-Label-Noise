# Framework

This component consists of a Machine Learning pipeline receiving YAML configs and outputting CSV files into ```./output/results```

## Start an Experiment

- Start the Pipeline with ```python3 main.py <config_folder>```
- The results are saved under ```./output/results```


## Start an Hyperparameter Optimization

- Start the HP Optimizer with ```python3 hp_tuning.py <config_folder>```
- The final hp configs are saved under ```./output/hpyamls```
- The detailed accuracy for each hp constellation is saved under ```./output/hpresults```

## Logging

The logs are saved under ```./logs```
