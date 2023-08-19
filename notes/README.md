


## experiment log


- baseline-plus-augms vs baseline: augmentations are helping. Overfitting occurs much later with augmentations.


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


