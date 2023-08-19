


## experiment log



- baseline-plus-augms-again-ft1 (vs baseline-plus-augms-again): SWA, continuation from baseline-plus-augms-again. Results are basically the same.

```
train_loss                    0.21057
valid_loss                   0.220207
multi_label_accuracy         0.911513
multilabel_hamming_loss      0.088487
macro_f1_score_multilabel    0.641824
```

- baseline-plus-augms-again (vs baseline): augmentations are helping. Overfitting occurs much later with augmentations.
```
train_loss                   0.208483
valid_loss                   0.219635
multi_label_accuracy          0.91159
multilabel_hamming_loss       0.08841
macro_f1_score_multilabel    0.641377
```


- baseline
```
train_loss                   0.226831
valid_loss                   0.236417
multi_label_accuracy         0.901664
multilabel_hamming_loss      0.098336
macro_f1_score_multilabel    0.606951
```

### note 

baseline and baseline-plus-augms have had a bug in dataloader, where the labels were shuffled, so they don't have reproducible results. This has been fixed, and in "baseline-plus-augms-again" the results are reproducible. Check the commit after "77040d63210906f06968016dbf3de44131923c28" for the fix. 




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


