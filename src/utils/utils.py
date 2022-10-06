import os
import hydra
import mlflow
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote, urlparse

import torch
import torch.backends.cudnn as cudnn

from .resource import MlflowWriter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cudnn():
    cudnn.benchmark = False
    cudnn.deterministic = True


def set_mlflow_hydra(cfg):
    cwd = hydra.utils.get_original_cwd()
    path = Path(cwd)
    mlflow.set_tracking_uri("file://" + cwd + "/mlruns")
    EXPERIMENT_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = MlflowWriter(EXPERIMENT_NAME)
    for key, value in cfg.items():
        print(f"{key}: {value}")
    return path, writer


def make_run(trial, cfg, writer):
    tags = {'trial': trial}
    writer.create_new_run(tags)
    writer.log_params_from_omegaconf_dict(cfg)
    arti_path = writer.run.info.artifact_uri
    arti_path = unquote(urlparse(arti_path).path)
    return arti_path


def save_result(train_acc, test_acc, writer, elapsed):
    writer.log_metric("train_acc_final", train_acc)
    writer.log_metric("test_acc_final", test_acc)
    writer.log_param("this_fold_time", elapsed)
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    writer.set_terminated()
