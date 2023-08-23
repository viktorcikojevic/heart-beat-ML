


## experiment log

- baseline-plus-augms-manymodels: ensemble of 4 folds and 2 different training schemes (8 models).




- baseline-plus-weightedsampler (vs baseline): results are similar to the baseline, a bit worse.


```
epoch                               2
train_loss                   0.399473
valid_loss                   0.376508
multi_label_accuracy         0.837529
multilabel_hamming_loss      0.162471
macro_f1_score_multilabel    0.576434
```

- baseline-plus-augms (vs baseline): augmentations are helping. Overfitting occurs much later with augmentations.


```
epoch                              46
train_loss                   0.271569
valid_loss                   0.349168
multi_label_accuracy         0.856433
multilabel_hamming_loss      0.143567
macro_f1_score_multilabel    0.620491
```


- baseline

```
epoch                               4
train_loss                   0.323445
valid_loss                   0.380388
multi_label_accuracy         0.843289
multilabel_hamming_loss      0.156712
macro_f1_score_multilabel    0.564015
```




## useful papers & links related to ECG

- useful visualizations: https://physionet.org/lightwave/?db=ptb-xl/1.0.2
- baselines: https://arxiv.org/abs/2004.13701


CHECK FOR MORE PAPERS !!!

## useful timeseries libs

- https://github.com/uber/orbit from uber, nice thing is that it's bayesian, so you can do probabilistic shenanigans with it
- https://github.com/lmmentel/awesome-time-series glossary
- https://github.com/cure-lab/Awesome-time-series papers
- https://github.com/facebook/prophet facebook stuffs
- https://ts.gluon.ai/stable/ another ilb
- https://github.com/unit8co/darts another one


