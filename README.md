






- check the notes [here](notes/README.md)
- EDA is performed [here](notebooks/eda.ipynb)
- dataloaders are [here](tools/dataloaders.py)

# Training


- To perform training, run the following command:

```bash
python train.py --config-file config.json
```

- Training script meant to reproduce the results: [here](train.sh)

# TODO

- [ ] Fix the `train_on_ffts` flag in dataloader: you're outputing the complex array.