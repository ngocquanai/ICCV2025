
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from torchmetrics.classification import CalibrationError, MulticlassCalibrationError
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy
from argparse import ArgumentParser
from utils import *
import numpy as np
import psutil
import time

from models import vision_transformer_fclayer

# from logger import create_logger


def train(config, model, criterion, dl, opt, scheduler, logger, epoch, task):

    model.train()
    model = model.cuda()

    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        start_time = time.time()
        for i, batch in enumerate(dl):
            # torch.cuda.empty_cache()
            if task == 'vtab':
                x, y = batch[0].cuda(), batch[1].cuda()
            elif task == 'fgvc':
                if not isinstance(batch["image"], torch.Tensor):
                    for k, v in batch.items():
                        data[k] = torch.from_numpy(v)
                x = batch["image"].float().cuda()
                y = batch["label"].cuda()
            else:
                print("Error Task Name")
                break
            out = model(x)

            # loss = F.cross_entropy(out, y)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step(ep)
        
        ram_used = psutil.virtual_memory().used / (1024.0 * 1024.0)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


        end_time = time.time()
        epoch_time = end_time - start_time

        if ep <= 10 and ep%2 == 0:

            logger.info('RAM used: '+str(int(ram_used*1000)/1000)+' memory: '+str(int(memory_used*1000) /1000)+'MB' + " TIME: " + str(int(epoch_time*1000) / 1000))
            print('RAM used: '+str(int(ram_used*1000)/1000)+' memory: '+str(int(memory_used*1000) /1000)+'MB' + "  TIME: " + str(int(epoch_time*1000) / 1000))

        if ep % 5 == 4:
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            acc, ece = test(model, test_dl, task)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                print(acc)
                logger.info('-'*50 + f" Current best acc: {acc} " + '-'*50)
                # save('vit_sct', config['task'], config['name'], model, acc, ep)

                ece =  int(ece*1000)/1000
                config['best_ece'] = ece
                print("ECE: ", ece)
                logger.info('-'*50 + f" Current best ECE: {ece} " + '-'*20)


            logger.info(str(ep)+' '+str(acc)+' memory: '+str(memory_used)+'MB')
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl, task):
    model.eval()
    model = model.cuda()
    
    # Initialize metrics
    all_preds = []
    all_targets = []
    all_probs = []
    
    for batch in dl:
        torch.cuda.empty_cache()
        if task == 'vtab':
            x, y = batch[0].cuda(), batch[1].cuda()
        elif task == 'fgvc':
            if not isinstance(batch["image"], torch.Tensor):
                data = {}
                for k, v in batch.items():
                    data[k] = torch.from_numpy(v)
                batch = data
            x = batch["image"].float().cuda()
            y = batch["label"].cuda()
        
        # Forward pass
        logits = model(x)
        
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get predictions
        preds = torch.argmax(probs, dim=1)
        
        # Collect for later computation
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        all_probs.append(probs.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    
    # Calculate accuracy
    correct = (all_preds == all_targets).sum().item()
    total = all_targets.size(0)
    accuracy = correct / total
    
    # Calculate ECE using torchmetrics
    num_classes = all_probs.size(1)


    ece_metric = MulticlassCalibrationError(
        num_classes=num_classes, 
        n_bins=15,  # You can adjust the number of bins as needed
        norm='l1'   # Using L1 norm (standard ECE)
    )
    ece = ece_metric(all_probs, all_targets)
    
    return accuracy, ece.item()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--eval', type=str, default='True')
    parser.add_argument('--dpr', type=float, default=0.1)
    parser.add_argument('--topN', type=int, default=None)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k_lora')
    parser.add_argument('--model_checkpoint', type=str, default='./released_models/ViT-B_16.npz')
    parser.add_argument('--model_type', type=str, default='vit_lora')
    parser.add_argument('--task', type=str, default='vtab')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--tuning_mode', type=str, default='lora')

    parser.add_argument('--config_folder', type=str, default='configs')
    parser.add_argument('--output', type=str, default='output')


    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    config = get_config('model_lora', args.task, args.dataset, args.config_folder)

    if args.topN is not None:
        topN = args.topN
    else:
        topN = config['topN']

    exp_base_path = './%s/%s/%s/%s'%(args.output, args.model_type, args.task, config['name']+'_dim_%d'%(topN))
    mkdirss(exp_base_path)
    logger = create_logger(log_path=exp_base_path, log_name='training')

    logger.info(args)
    logger.info(config)

    ## prepare training data
    if args.eval == 'True':
        evalflag = True
    else:
        evalflag = False

    if 'train_aug' in config.keys():
        train_aug = config['train_aug']
    else:
        train_aug = False
    
    if args.task == 'vtab':
        from dataloader.vtab import *
        basedir = '../../datasets/vtab-1k'
        train_dl, test_dl = get_data(basedir, args.dataset, logger, evaluate=evalflag, train_aug=train_aug, batch_size=config['batch_size'])
        print(args.model_checkpoint, "-"*100)
    elif args.task == 'fgvc':
        from dataload.loader import construct_train_loader, construct_test_loader
        train_dl = construct_train_loader(args.dataset, batch_size=config['batch_size'])
        test_dl = construct_test_loader(args.dataset, batch_size=config['batch_size'])
        print(len(train_dl), len(test_dl))


    if 'swin' in args.model:
        model = create_model(args.model, pretrained=False, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode, topN=topN)
        model.load_state_dict(torch.load(args.model_checkpoint)['model'], False) ## not include adapt module
    else:
        model = create_model(args.model, checkpoint_path=args.model_checkpoint, drop_path_rate=args.dpr, tuning_mode=args.tuning_mode)

    model.reset_classifier(config['class_num'])    
    
    logger.info(str(model))

    config['best_acc'] = 0
    config['best_ece'] = 100000
    config['task'] = args.task

    trainable = []
    for n, p in model.named_parameters():
        if 'head' in n:
            trainable.append(p)
            logger.info(str(n))
        else:
            p.requires_grad = False

    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    if 'cycle_decay' in config.keys():
        cycle_decay = config['cycle_decay']
    else:
        # default 0.1
        cycle_decay = 0.1

    scheduler = CosineLRScheduler(opt, t_initial=config['epochs'],
                                  warmup_t=config['warmup_epochs'], lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=cycle_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of extra params:{}M".format(n_parameters/1000000))

    logger.info(f"number of extra params: {n_parameters}")

    if config['custom_loss'] == True :
        criterion = HieraCrossEntropy(dataset= args.dataset)
        logger.info("Hierarchical Cross Entropy")
        print("Hierarchical Cross Entropy")
        
    elif config['labelsmoothing'] > 0.:
        ## label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=config['labelsmoothing'])
        logger.info('label smoothing')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        logger.info('CrossEntropyLoss')
    
    model = train(config, model, criterion, train_dl, opt, scheduler, logger, config['epochs'], args.task)
    print(config['best_acc'])

    logger.info('end')