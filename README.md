



# Repo structure


```
experiments
├── baseline
├── baseline-plus-augms
├── baseline-plus-augms-manymodels
└── baseline-plus-weightedsampler
src
├── dataloaders.py
├── fastai_fix.py
├── __init__.py
├── loss.py
├── metrics.py
├── models.py
└── utils.py
notebooks/
├── correlation-analysis.ipynb
├── debug.ipynb
├── eda.ipynb
├── fft-viz.ipynb
├── generate_dataset.ipynb
├── heart-beat-speed.ipynb
└── qrs_detection.ipynb
```

# Training


- To perform training, run the following command:

```bash
python train.py --config-file config.json
```

- Training script meant to reproduce the results: [here](train.sh)

# TODO

- [ ] Fix the `train_on_ffts` flag in dataloader: you're outputing the complex array.