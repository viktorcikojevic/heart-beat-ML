import pandas as pd
from src.fastai_fix import *
from tqdm.notebook import tqdm
from src.dataloaders import (
    ECGDataset, 
    DeviceDataLoader,
)
from fastxtend.vision.all import EMACallback
from tqdm import tqdm
from src.utils import seed_everything, WrapperAdamW
import argparse
import json
import torch
import os
import sys
from src.models import DeepHeartModel
from pdb import set_trace
from src.loss import binary_cross_entropy
from src.metrics import *

def read_config_file(file_path):
    with open(file_path, "r") as f:
        config_data = json.load(f)
    
    experiment_dir = config_data["OUT"]    
    
    # copy the config file to the experiment directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    print(f"Copying config file to experiment directory: {experiment_dir}")
    try:
        os.system(f"cp {file_path} {experiment_dir}")
    except:
        raise Exception(f"Could not copy config file to experiment directory: {experiment_dir}")
    
    return config_data


def train(config):
    
    # Create train and validation datasets and dataloaders
    
    ds_train = ECGDataset(
        config["PATH"],
        mode="train",
        L=config["LMAX"],
        test_folds=config["TEST_FOLDS"],
    )

    dl_train = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_train,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS_DATALOADER"],
            persistent_workers=True,
        )
    )

    ds_val = ECGDataset(
        config["PATH"],
        mode="val",
        L=config["LMAX"],
        test_folds=config["TEST_FOLDS"],
    )

    dl_val = DeviceDataLoader(
        torch.utils.data.DataLoader(
            ds_val,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS_DATALOADER"],
            persistent_workers=True,
        )
    )

    data = DataLoaders(dl_train, dl_val)
    print("Dataloaders created.")
    
    
    model = config["MODEL"](**config["MODEL_KWARGS"])
    if config["WEIGHTS"]:
        print("Loading weights from ...", config["WEIGHTS"])
        model.load_state_dict(torch.load(config["WEIGHTS"]))
    
    # print number of million parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    model = nn.DataParallel(model)
    model = model.cuda()
    cbs = [
        GradientClip(3.0),
        CSVLogger(),
        SaveModelCallback(monitor='valid_loss', comp=np.less, every_epoch=True),
        GradientAccumulation(n_acc=config["BATCH_SIZE_EFFECTIVE"] // config["BATCH_SIZE"]),
    ]
    if config["EMA"]:
        cbs.append(EMACallback())

    learn = Learner(
        data,
        model,
        cbs=cbs,
        path=config["OUT"],
        loss_func=config["LOSS_FUNC"],
        metrics=config["METRICS"],
        opt_func=partial(WrapperAdamW, eps=1e-7),
    ).to_fp16()

    # fit using the 1cycle policy
    
    
    learn.fit_one_cycle(
        n_epoch=config["EPOCHS"],
        lr_max=config["LR_MAX"],
        wd=0.05,
        pct_start=0.01,
        div=config["DIV"],
        div_final=config["DIV_FINAL"],
        moms=tuple(config["MOMS"]) if config["MOMS"] else None
    )
    
    
    
    


def main():
    
    
    
    parser = argparse.ArgumentParser(
        description="Create a model from a JSON config file."
    )
    parser.add_argument("--config-file", type=str, help="Path to the JSON config file.", required=True)
    parser.add_argument(
        "configs",
        nargs="*",
        metavar=("KEY", "VALUE"),
        help="The JSON config key to override and its new value.",
    )

    args = parser.parse_args()
    config_file_path = args.config_file

    config_data = read_config_file(config_file_path)
    
    

    if args.configs:
        for config_key, config_value in zip(args.configs[::2], args.configs[1::2]):
            keys = config_key.split(".")
            last_key = keys.pop()

            current_data = config_data
            for key in keys:
                current_data = current_data[key]

            try:
                value = json.loads(config_value)
            except json.JSONDecodeError:
                value = config_value

            current_data[last_key] = value

    print("Training with the following configuration:")
    print(json.dumps(config_data, indent=4))
    print("_______________________________________________________")
    
    config_data["MODEL"] = getattr(sys.modules[__name__], config_data["MODEL"])
    
    if config_data["LOSS_FUNC"] == "binary_cross_entropy":
        config_data["LOSS_FUNC"] = binary_cross_entropy
    else:
        config_data["LOSS_FUNC"] = getattr(sys.modules[__name__], config_data["LOSS_FUNC"])
        
    # config_data["METRICS"] = binary_cross_entropy
    config_data["METRICS"] = [
        getattr(sys.modules[__name__], cfg) for cfg in config_data["METRICS"]
    ]
        
    seed_everything(config_data["SEED"])
    os.makedirs(config_data["OUT"], exist_ok=True)
    train(config_data)


if __name__ == "__main__":
    main()