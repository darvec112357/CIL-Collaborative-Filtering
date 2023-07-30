# RainFM: A Stratified Hybrid Model for Enhanced Predictive Accuracy in Recommendation Systems

Group: Recommenders  
Group members: Rainer Feichtinger, Rongxing Liu, Justin Lo, Ruben Schenk  

## Dependencies
All available packages can be installed via:
```
pip install -r requirements.txt
```

## Training
To organise the experiments, we have compiled them into a jupyter notebook [here](experiments.ipynb). The notebook contains the experiments that we have conducted with their respective hyperparameters after they have been tuned via cross-validation. There is a short explanation provided for each method, but further details can be read from our report. 

In the jupyter notebook, the experiments were conducted and evaluated on our train-test split. To produced the output submitted to kaggle, one may change the training methods to utilise the 'train_df_full' Dataframe instead of the 'train_df' Dataframe. The results can then be exported as required.

## Results
Below are the results of the experiments conducted on our train-test split. 

| Method                  | Test RMSE | Test MAE |
| :---------------------- | :-------: |:-------: |
| Item Average            | 1.0309    | 0.8398  |
| I-SVD with ALS          | 0.9921    | 0.7896  |
| Generalized MF          | 1.0822    | 0.8795  |
| Multi-Layer Perceptron  | 1.0029    | 0.8105  |
| NeuFM (Pretrained)      | 1.0041    | 0.8092  |
| BFM Baseline            | 0.9777    | 0.7809  |
| BFM (augmented with KNN)| 0.9840    | 0.7808  |
| RainFM                  |**0.9695** | **0.7714**|