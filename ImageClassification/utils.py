import os
import torch
import random
import numpy as np
import yaml
import logging
from pathlib import Path

# HICE
import torch.nn as nn
import torch.nn.functional as F
from taxonomy.cifar100_config import CIFAR100, CIFAR100_TAXONOMY
from taxonomy.caltech101 import CALTECH101, CALTECH101_TAXONOMY
from taxonomy.pets import PETS, PETS_TAXONOMY
from taxonomy.resisc45 import RESISC45, RESISC45_TAXONOMY
from taxonomy.sun397 import SUN397, SUN397_TAXONOMY
from taxonomy.taxonomy import get_layers_weight


def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def save(model_type, task, name, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'sct_mlp' in n or 'sct_mlp' in n or 'head' in n or 'q_l' in n or 'k_l' in n or 'v_l' in n:
            trainable[n] = p.data

    torch.save(trainable, '../output')
    

def load(model_type, task, name, model):
    model = model.cpu()
    st = torch.load('../output/%s/%s/%s/ckpt_epoch_best.pt'%(model_type, task, name))
    model.load_state_dict(st, False)
    return model

def get_config(model_type, task, dataset_name, folder):
    with open('./%s/%s/%s/%s.yaml'%(folder, model_type, task, dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config



def create_logger(log_path, log_name):
    """
    Creates a logger to log messages to a file.

    :param log_path: The path where the log file should be saved.
    :param log_name: The name of the log file.
    :return: A logger instance.
    """
    # Create the directory if it doesn't exist
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)  # Set the logging level to INFO
    
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(f"{log_path}/{log_name}.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger

#################################  HICE  #####################################


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()





class HieraCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, dataset, alpha=0.1):
        super(HieraCrossEntropy, self).__init__()
        self.loss_weight= [alpha, alpha, alpha, 1 - 3*alpha]
        if dataset == "cifar100" :
            dataset_name = CIFAR100
            dataset_taxonomy = CIFAR100_TAXONOMY
        elif dataset == "caltech101" :
            dataset_name = CALTECH101
            dataset_taxonomy = CALTECH101_TAXONOMY
        elif dataset == "oxford_iiit_pet" :
            dataset_name = PETS
            dataset_taxonomy = PETS_TAXONOMY
        elif dataset == "resisc45" :
            dataset_name = RESISC45
            dataset_taxonomy = RESISC45_TAXONOMY
        elif dataset == "sun397" :
            dataset_name = SUN397
            dataset_taxonomy = SUN397_TAXONOMY
        else :
            print("ERROR")

        hidden_W = get_layers_weight(dataset_name, dataset_taxonomy)
        self.W0 = hidden_W[0].cuda()
        self.W1 = hidden_W[1].cuda()
        self.W2 = hidden_W[2].cuda()
        self.ce = nn.CrossEntropyLoss()

        print(self.W0.shape, self.W1.shape, self.W2.shape)
        print("USING LOSS WEIGHT HEREEE: ", self.loss_weight)


    def transform_labels(self, target, W) :

        # input: argmax type
        target_onehot = torch.nn.functional.one_hot(target, num_classes= W.shape[1]).float()
        target_transformed = torch.matmul(target_onehot, W.T)

        return torch.argmax(target_transformed, axis=1) # Return in argmax type again


    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = F.cross_entropy(x, target)
        main_loss = loss

        targets_layer2 = self.transform_labels(target, self.W2)
        logits_layer2 = torch.matmul(x, self.W2.T)
        # print(targets_layer2, logits_layer2.shape)
        loss_layer2 = F.cross_entropy(logits_layer2, targets_layer2)

        targets_layer1 = self.transform_labels(targets_layer2, self.W1)
        logits_layer1 = torch.matmul(logits_layer2, self.W1.T)
        # print(targets_layer1, logits_layer1.shape)
        loss_layer1 = F.cross_entropy(logits_layer1, targets_layer1)

        targets_layer0 = self.transform_labels(targets_layer1, self.W0)
        logits_layer0 = torch.matmul(logits_layer1, self.W0.T)
        # print(targets_layer0, logits_layer0.shape)
        loss_layer0 = F.cross_entropy(logits_layer0, targets_layer0)

        total_loss = self.loss_weight[0] * loss_layer0 + self.loss_weight[1] * loss_layer1 + self.loss_weight[2] * loss_layer2 + self.loss_weight[3] * main_loss
        # print(self.loss_weight)
        return total_loss

