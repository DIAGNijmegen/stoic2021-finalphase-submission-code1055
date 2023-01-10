import numpy as np
import torch
import torch.nn as nn
#import monai
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from segmentation.segconfig.segmodelconfig import get_model, get_modelconfig
from segmentation.data.datasets import SupervisedDataset, UnsupervisedDataset_stoic, ValidationDataset, UnsupervisedDataset_tcia, MosMedDataset, HustDataset
from segmentation.data.transforms3D import ToTensor3D, Normalize3D, elastic_wrapper, elastic_deform_robin
from segmentation.data.transforms3D import get_transform, Zoom
from segmentation.data.transforms3D import Compose as My_Compose
from segmentation.data.transforms3D import Identity as My_Identity
from segmentation.segconfig.segconfig import MultiConfig
import nibabel as nib
import scipy.ndimage as ndimage
import random
import time
import argparse
from segmentation.segutils import DiceCELoss, diceloss, dice_score_fn, MyBCELoss, saveModel, load_model, IoU, update_teacher, set_deterministic, seed_worker
import logging
import torch.utils.data as data
from tqdm import tqdm, trange
from config.dataconfig import get_dataset
import torch.nn.functional as F
import evaluate as ev
from data.mosmed import get_mosmed_dataset
from data.hust import get_hust_dataset
from config.dataconfig import MosmedConfig, GPUDeformedDataConfig, HustConfig
import sys
from loss import BinaryCrossEntropySevOnly, CrossEntropySevOnly
from segmentation.data.transforms3D import elastic_deform_robin

def parse_args():
    parser = argparse.ArgumentParser(description='Train STOIC network')
    # general
    parser.add_argument('--gpu',
                        default="6",
                        help='gpu id',
                        type=str)
    parser.add_argument('--cv',
                        help='Use cross-validation',
                        dest='cv',
                        action='store_true')
    parser.add_argument('--no-cv',
                        dest='cv',
                        action='store_false')
    parser.set_defaults(cv=False)
    parser.add_argument('--nick',
                        default='',
                        help='Prepend a nickname to the output directory',
                        type=str)
    parser.add_argument('--mode',
                        default='1',
                        help='1=onlySupervised, 2=MeanTeacher_withoutAugmentation, 3=MeanTeacher_elasticDeformation',
                        type=str)
    parser.add_argument('--imsize',
                        default='256',
                        help='image size: 256 or 128',
                        type=str)
    args = parser.parse_args()
    args.model = 'multinext'
    return args


def run(config, args):
    model = get_model(config)
    teacher = get_model(config)
    modelconfig = config.modelconfig
    model = model.to(config.DEVICE)
    teacher = teacher.to(config.DEVICE)
    with torch.no_grad():
        for name, param in teacher.named_parameters():
            model.state_dict()[name]
            param.data = model.state_dict()[name]

    assert config.MODEL_NAME == 'multinext'
    model_name = 'convnextransformer' if modelconfig.use_transformer else 'convnext'
    if hasattr(modelconfig, 'size') and modelconfig.size == 'small':
        model_name += 'Small'
    elif hasattr(modelconfig, 'size') and modelconfig.size == 'base':
        model_name += 'Base'
    elif hasattr(modelconfig, 'size') and modelconfig.size == 'micro':
        model_name += 'Micro'
    #TODO use better identifier
    identifier = ''.join((model_name))
    identifier = ''.join((args.nick, '_', identifier, '_', config.datetime))

    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_tcia = Normalize3D(mean=0.2268, std=0.3048) if config.DO_NORMALIZATION else My_Identity()
    normalize_stoic = Normalize3D(mean=0.3110, std=0.3154) if config.DO_NORMALIZATION else My_Identity()
    normalize_mosmed = Normalize3D(mean=0.2926, std=0.3119) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_seg_val = My_Compose([Zoom([zoomsize, zoomsize, zoomsize]), ToTensor3D(), normalize_tcia])
    transform_seg_train = My_Compose([get_transform(config.IMAGE_SIZE, normalize_tcia), ToTensor3D()])


    #TODO add random number to SupervisedDataset to do cross validation for segmentation too
    train_dataset_seg = SupervisedDataset(path=config.DATA_PATH_SEG, transform=transform_seg_train, train=True)
    my_val_dataset_seg = SupervisedDataset(path=config.DATA_PATH_SEG, transform=transform_seg_val, train=False)
    val_dataset_unlabeled = ValidationDataset(path=config.DATA_PATH_SEG, transform=transform_seg_val)
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader_seg = torch.utils.data.DataLoader(my_val_dataset_seg, batch_size=config.Batch_SIZE,
                                                num_workers=config.WORKERS,
                                                shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

    loader_cls = data.DataLoader
    datasets = {}
    for phase, cfg in config.dataconfigs.items():
        cfg.do_normalization = config.DO_NORMALIZATION
        datasets[phase] = get_dataset(cfg)
    #datasets = {phase: get_dataset(cfg) for phase, cfg in config.dataconfigs.items()}
    train_loader_cls = loader_cls(
        dataset=datasets["train"], num_workers=config.WORKERS, batch_size=config.Batch_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker
    )
    val_loader_cls = data.DataLoader(
        dataset=datasets["val"], num_workers=config.WORKERS, batch_size=config.Batch_SIZE,
        worker_init_fn=seed_worker
    )

    train_dataset_mos = MosMedDataset(config, mode='train')
    val_dataset_mos = MosMedDataset(config, mode='val')
    train_loader_mos = data.DataLoader(
        dataset=train_dataset_mos,
        num_workers=config.WORKERS,
        batch_size=config.Batch_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker
    )
    val_loader_mos = data.DataLoader(
        dataset=val_dataset_mos,
        num_workers=config.WORKERS,
        batch_size=config.Batch_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker
    )

    train_dataset_hust = HustDataset(config, mode='train')
    val_dataset_hust = HustDataset(config, mode='val')
    train_loader_hust = data.DataLoader(
        dataset=train_dataset_hust,
        num_workers=config.WORKERS,
        batch_size=config.Batch_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker
    )
    val_loader_hust = data.DataLoader(
        dataset=val_dataset_hust,
        num_workers=config.WORKERS,
        batch_size=config.Batch_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker
    )

    train_loader_seg_iter = iter(train_loader_seg)
    train_loader_cls_iter = iter(train_loader_cls)
    train_loader_mos_iter = iter(train_loader_mos)
    train_loader_hust_iter = iter(train_loader_hust)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.modelconfig.learning_rate, weight_decay=config.modelconfig.weight_decay)

    if config.LOSS_SEG == 'dicece':
        loss_fn_seg = DiceCELoss()
    elif config.LOSS_SEG == 'balancedce':
        loss_fn_seg = MyBCELoss(device=config.DEVICE)
    else:
        loss_fn_seg = torch.nn.BCEWithLogitsLoss()

    loss_fn_cls = CrossEntropySevOnly(pos_weight=None) if config.LOSS_CLS == 'ce' else BinaryCrossEntropySevOnly(pos_weight=1.0)

    #TODO adjust identifier and tensorboard for cross validation
    if config.cv_enabled:
        foldname = 'cv' + str(config._current_fold)
        identifier = os.path.join(config.LOGS_PATH, 'logs', identifier, foldname)
    else:
        identifier = os.path.join(config.LOGS_PATH, 'logs', identifier)
    os.makedirs(identifier, exist_ok=True)
    writer = SummaryWriter(identifier)

    iteration, epoch_cls, epoch_mos, epoch_seg, epoch_hust, epoch = 0, 0, 0, 0, 0, 0
    print('Epoch ', epoch_cls)
    while epoch < config.MAX_EPOCHS:
        optimizer.zero_grad()
        # segmentation TCIA
        if 'tcia' in config.datasets:
            input, label = next(train_loader_seg_iter, ('end', None))
            if input == 'end':
                train_loader_seg_iter = iter(train_loader_seg)
                input, label = next(train_loader_seg_iter, ('end', None))
                if config.datasets[0] == 'tcia' or epoch_seg % 5 == 0:
                    iou, dicescore, loss_sum = validation_seg(teacher, my_val_loader_seg, config, loss_fn_seg)
                    writer.add_scalar('Validation/IOU ema', iou, epoch_seg)
                    writer.add_scalar('Validation/Dice Score ema', dicescore, epoch_seg)
                    writer.add_scalar('Validation/validation loss tcia ema', loss_sum, epoch_seg)
                epoch_seg += 1
                if config.datasets[0] == 'tcia':
                    saveModel(model, teacher, optimizer, epoch, config, identifier) #Epoch is the same epoch as in validation score
                    epoch = epoch_seg
                    if config.change_loss_weight:
                        update_datasetweights(config, epoch)
                    print('Epoch ', epoch)
            loss_seg, loss_seg_sum, loss_grad_sum = segmentation_step_tcia(input, label, model, config, loss_fn_seg)
            writer.add_scalar('Training/train loss tcia', loss_seg_sum, iteration)
            writer.add_scalar('Gradient/grad tcia', loss_grad_sum, iteration)
            #print('train loss tcia', loss_seg_sum)
        else:
            loss_seg_sum = 0
            epoch_seg = config.MAX_EPOCHS

        # classification STOIC
        if 'stoic' in config.datasets:
            v_tensor, age, sex, inf_gt, sev_gt = next(train_loader_cls_iter, ('end', None, None, None, None))
            if v_tensor == 'end':
                train_loader_cls_iter = iter(train_loader_cls)
                v_tensor, age, sex, inf_gt, sev_gt = next(train_loader_cls_iter, ('end', None))
                if config.datasets[0] == 'stoic' or epoch_cls % 2 == 0:
                    auc_sev2, loss_sum = validation_stoic(val_loader_cls, model, config, loss_fn_cls)
                    auc_sev2_ema, loss_sum_ema = validation_stoic(val_loader_cls, teacher, config, loss_fn_cls)
                    writer.add_scalar('Validation/validation loss stoic', loss_sum, epoch_cls)
                    writer.add_scalar('Validation/auc_sev2 stoic', auc_sev2, epoch_cls)
                    writer.add_scalar('Validation/validation loss stoic ema', loss_sum_ema, epoch_cls)
                    writer.add_scalar('Validation/auc_sev2_ema stoic', auc_sev2_ema, epoch_cls)
                epoch_cls += 1
                if config.datasets[0] == 'stoic':
                    saveModel(model, teacher, optimizer, epoch, config, identifier) #Epoch is the same epoch as in validation score
                    epoch = epoch_cls
                    if config.change_loss_weight:
                        update_datasetweights(config, epoch)
                    print('Epoch ', epoch)
            loss_cls, loss_cls_sum, loss_grad_sum = classification_step_stoic((v_tensor, age, sex), (inf_gt, sev_gt), model, config, loss_fn_cls)
            writer.add_scalar('Training/train loss stoic', loss_cls_sum, iteration)
            writer.add_scalar('Gradient/grad stoic', loss_grad_sum, iteration)
            #print('train loss stoic', loss_cls_sum)
        else:
            loss_cls_sum = 0
            epoch_cls = config.MAX_EPOCHS

        # classification MosMed
        if 'mosmed' in config.datasets:
            v_tensor, age, sex, inf_gt, sev_gt = next(train_loader_mos_iter, ('end', None, None, None, None))
            if v_tensor == 'end':
                train_loader_mos_iter = iter(train_loader_mos)
                v_tensor, age, sex, inf_gt, sev_gt = next(train_loader_mos_iter, ('end', None, None, None, None))
                if config.datasets[0] == 'mosmed' or epoch_mos % 2 == 0:
                    auc_sev2_ema, loss_sum_ema = validation_mos(val_loader_mos, teacher, config, loss_fn_cls)
                    writer.add_scalar('Validation/validation loss mosmed ema', loss_sum_ema, epoch_mos)
                    writer.add_scalar('Validation/auc_sev2_ema mosmed', auc_sev2_ema, epoch_mos)
                epoch_mos += 1
                if config.datasets[0] == 'mosmed':
                    saveModel(model, teacher, optimizer, epoch, config, identifier) #Epoch is the same epoch as in validation score
                    epoch = epoch_mos
                    if config.change_loss_weight:
                        update_datasetweights(config, epoch)
                    print('Epoch ', epoch)
            loss_mos, loss_mos_sum, loss_grad_sum = classification_step_mos((v_tensor, age, sex), (inf_gt, sev_gt), model, config, loss_fn_cls, iteration)
            writer.add_scalar('Training/train loss mosmed', loss_mos_sum, iteration)
            writer.add_scalar('Gradient/grad mosmed', loss_grad_sum, iteration)
            #print('train loss mosmed', loss_mos_sum)
        else:
            loss_mos_sum = 0
            epoch_mos = config.MAX_EPOCHS

        # classification HUST
        if 'hust' in config.datasets:
            v_tensor, age, sex, inf_gt, sev_gt = next(train_loader_hust_iter, ('end', None, None, None, None))
            if v_tensor == 'end':
                train_loader_hust_iter = iter(train_loader_hust)
                v_tensor, age, sex, inf_gt, sev_gt = next(train_loader_hust_iter, ('end', None))
                if config.datasets[0] == 'hust' or epoch_hust % 3 == 0:
                    auc_sev2_ema, loss_sum_ema = validation_hust(val_loader_hust, teacher, config, loss_fn_cls)
                    writer.add_scalar('Validation/validation loss hust ema', loss_sum_ema, epoch_hust)
                    writer.add_scalar('Validation/auc_sev2_ema hust', auc_sev2_ema, epoch_hust)
                epoch_hust += 1
                if config.datasets[0] == 'hust':
                    saveModel(model, teacher, optimizer, epoch, config,
                              identifier)  # Epoch is the same epoch as in validation score
                    epoch = epoch_hust
                    if config.change_loss_weight:
                        update_datasetweights(config, epoch)
                    print('Epoch ', epoch)
            loss_hust, loss_hust_sum, loss_grad_sum = classification_step_hust((v_tensor, age, sex),
                                                                              (inf_gt, sev_gt), model, config,
                                                                              loss_fn_cls)
            writer.add_scalar('Training/train loss hust', loss_hust_sum, iteration)
            writer.add_scalar('Gradient/grad hust', loss_grad_sum, iteration)
            # print('train loss stoic', loss_cls_sum)
        else:
            loss_hust_sum = 0
            epoch_hust = config.MAX_EPOCHS

        loss_sum = (loss_cls_sum + loss_seg_sum + loss_mos_sum + loss_hust_sum)/ len(config.datasets)
        writer.add_scalar('Training/train loss', loss_sum, iteration)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        teacher = update_teacher(model, teacher, alpha=0.999)
        iteration += 1
        sys.stdout.flush()




def update_datasetweights(config, epoch):
    for i in range(1, len(config.dataset_weights)):
        if config.change_loss_weight:
            config.dataset_weights[i] = 1 - 1. / config.MAX_EPOCHS * epoch


def classification_step_stoic(inputs, labels, model, config, loss_fn):
    num_at_once = config.Batch_SIZE // config.NUM_STEPS
    v_tensor, age, sex, = inputs
    inf_gt, sev_gt = labels
    loss_sum, loss_grad_sum = 0, 0
    for i in range(0, v_tensor.shape[0], num_at_once):
        step = min(i + num_at_once, v_tensor.shape[0])

        v_ten, s_gt, i_gt = v_tensor[i:step], sev_gt[i:step], inf_gt[i:step]
        v_ten, s_gt, i_gt = v_ten.to(config.DEVICE), s_gt.to(config.DEVICE), i_gt.to(config.DEVICE)
        if config.DO_ELASTIC_DEFORM and random.random() <= config.deform_prob:
            alpha = random.random() * (config.deform_alpha[1] - config.deform_alpha[0]) + config.deform_alpha[0]
            sigma = random.randint(config.deform_sigma[0], config.deform_sigma[1])
            v_ten, __ = elastic_deform_robin(v_ten, None, alpha=alpha, sigma=sigma, device=config.DEVICE)
        if config.modelconfig.use_metadata:
            a, s = age[i:step], sex[i:step]
            a, s = a.to(config.DEVICE), s.to(config.DEVICE)
        else:
            a, s = None, None
        pred_stoic, pred_mos, pred_hust, pred_seg = model(v_ten, a, s, mode='stoic')
        pred_stoic.retain_grad()
        loss_weight = 1
        if config.change_loss_weight and config.datasets[0] != 'stoic':
            for i, dataset in enumerate(config.datasets):
                if dataset == 'stoic':
                    loss_weight = config.dataset_weights[i]
        loss = loss_fn(pred_stoic, i_gt.float(), s_gt.float()) / max((v_tensor.shape[0] // num_at_once), 1) / len(config.datasets) * loss_weight
        loss_sum += loss.item()
        loss.backward()
        loss_grad_sum += pred_stoic.grad
        pred_stoic = None
    return loss_sum, loss_sum, loss_grad_sum.norm()


def classification_step_hust(inputs, labels, model, config, loss_fn):
    num_at_once = config.Batch_SIZE // config.NUM_STEPS
    v_tensor, age, sex, = inputs
    inf_gt, sev_gt = labels
    loss_sum, loss_grad_sum = 0, 0
    for i in range(0, v_tensor.shape[0], num_at_once):
        step = min(i + num_at_once, v_tensor.shape[0])

        v_ten, s_gt, i_gt = v_tensor[i:step], sev_gt[i:step], inf_gt[i:step]
        v_ten, s_gt, i_gt = v_ten.to(config.DEVICE), s_gt.to(config.DEVICE), i_gt.to(config.DEVICE)
        if config.DO_ELASTIC_DEFORM and random.random() <= config.deform_prob:
            alpha = random.random() * (config.deform_alpha[1] - config.deform_alpha[0]) + config.deform_alpha[0]
            sigma = random.randint(config.deform_sigma[0], config.deform_sigma[1])
            v_ten, __ = elastic_deform_robin(v_ten, None, alpha=alpha, sigma=sigma, device=config.DEVICE)
        if config.modelconfig.use_metadata:
            a, s = age[i:step], sex[i:step]
            a, s = a.to(config.DEVICE), s.to(config.DEVICE)
        else:
            a, s = None, None
        pred_stoic, pred_mos, pred_hust, pred_se = model(v_ten, a, s, mode='hust')
        pred_hust.retain_grad()
        loss_weight = 1
        if config.change_loss_weight and config.datasets[0] != 'hust':
            for i, dataset in enumerate(config.datasets):
                if dataset == 'hust':
                    loss_weight = config.dataset_weights[i]
        loss = loss_fn(pred_hust, i_gt.float(), s_gt.float()) / max((v_tensor.shape[0] // num_at_once), 1) / len(config.datasets) * loss_weight
        loss_sum += loss.item()
        loss.backward()
        loss_grad_sum += pred_hust.grad
        pred_hust = None
    return loss_sum, loss_sum, loss_grad_sum.norm()


def classification_step_mos(inputs, labels, model, config, loss_fn, iteration):
    num_at_once = config.Batch_SIZE // config.NUM_STEPS
    v_tensor, age, sex, = inputs
    inf_gt, sev_gt = labels
    loss_sum, loss_grad_sum = 0, torch.zeros([1,], device=config.DEVICE)
    for i in range(0, v_tensor.shape[0], num_at_once):
        step = min(i + num_at_once, v_tensor.shape[0])

        v_ten, s_gt, i_gt = v_tensor[i:step], sev_gt[i:step], inf_gt[i:step]
        v_ten, s_gt, i_gt = v_ten.to(config.DEVICE), s_gt.to(config.DEVICE), i_gt.to(config.DEVICE)
        if config.DO_ELASTIC_DEFORM and random.random() <= config.deform_prob:
            alpha = random.random() * (config.deform_alpha[1] - config.deform_alpha[0]) + config.deform_alpha[0]
            sigma = random.randint(config.deform_sigma[0], config.deform_sigma[1])
            v_ten, __ = elastic_deform_robin(v_ten, None, alpha=alpha, sigma=sigma, device=config.DEVICE)
        pred_stoic, pred_mos, pred_hust, pred_seg = model(v_ten, None, None, mode='mosmed')
        pred_mos.retain_grad()
        loss_weight = 1
        if config.change_loss_weight and config.datasets[0] != 'mosmed':
            for i, dataset in enumerate(config.datasets):
                if dataset == 'mosmed':
                    loss_weight = config.dataset_weights[i]
        loss = loss_fn(pred_mos, i_gt.float(), s_gt.float()) / max((v_tensor.shape[0] // num_at_once), 1) / len(config.datasets) * loss_weight
        #If loss > 0.1 -> Fucking Mosmed fucked of
        if (loss * max((v_tensor.shape[0] // num_at_once), 1) < 0.1 or iteration < 100) or True:
            loss_sum += loss.item()
            loss.backward()
            loss_grad_sum = loss_grad_sum + pred_mos.grad
        pred_mos = None
    return loss_sum, loss_sum, loss_grad_sum.norm()


def segmentation_step_tcia(input, label, model, config, loss_fn):
    num_at_once = config.Batch_SIZE // config.NUM_STEPS
    input, label = input.to(config.DEVICE), label.to(config.DEVICE)
    if config.DO_ELASTIC_DEFORM and random.random() <= config.deform_prob:
        alpha = random.random() * (config.deform_alpha[1] - config.deform_alpha[0]) + config.deform_alpha[0]
        sigma = random.randint(config.deform_sigma[0], config.deform_sigma[1])
        input, label = elastic_deform_robin(input, label, alpha=alpha, sigma=sigma, device=config.DEVICE)
    loss_sum, loss_grad_sum = 0, 0
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp, lab = input[i:step], label[i:step]
        pred_stoic, pred_mos, pred_hust, pred_seg = model(inp, None, None, mode='tcia')
        pred_seg.retain_grad()
        loss_weight = 1
        if config.change_loss_weight and config.datasets[0] != 'tcia':
            for i, dataset in enumerate(config.datasets):
                if dataset == 'tcia':
                    loss_weight = config.dataset_weights[i]
        loss = loss_fn(pred_seg, lab) / max((label.shape[0] // num_at_once), 1) / len(config.datasets) * loss_weight
        loss_sum += loss.item()
        loss.backward()
        loss_grad_sum += pred_seg.grad
        pred_seg = None
    return loss, loss_sum, loss_grad_sum.norm()


def validation_stoic(data_loader, model, config, loss_fn):
    val_results = evaluate(data_loader, model, loss_fn, config, num_at_once = 1, mode='stoic')
    return val_results["auc_sev2"], val_results["loss"]

def validation_mos(data_loader, model, config, loss_fn):
    val_results = evaluate(data_loader, model, loss_fn, config, num_at_once=1, mode='mosmed')
    return val_results["auc_sev2"], val_results["loss"]

def validation_hust(data_loader, model, config, loss_fn):
    val_results = evaluate(data_loader, model, loss_fn, config, num_at_once=1, mode='hust')
    return val_results["auc_sev2"], val_results["loss"]

def validation_seg(model, data_loader, config, loss_fn):
    num_at_once = config.Batch_SIZE // config.NUM_STEPS
    model.eval()
    with torch.no_grad():
        loss, loss_ema = 0, 0
        up, down, up_ema, down_ema = 0, 0, 0, 0
        intersections, unions, intersections_ema, unions_ema = 0, 0, 0, 0
        for input, label in data_loader:
            input, label = input.to(config.DEVICE), label.to(config.DEVICE)
            for i in range(0, label.shape[0], num_at_once):
                step = min(i + num_at_once, label.shape[0])
                inp, lab = input[i:step], label[i:step]
                __, __, __, pred = model(inp, mode='tcia')
                loss += loss_fn(pred, lab) / max((label.shape[0] // num_at_once), 1) / len(data_loader)
                up_tmp, down_tmp = dice_score_fn(pred, lab)
                up += up_tmp.item()
                down += down_tmp.item()
                inter, uni = IoU(pred, lab, hard_label=True)
                intersections += inter.item()
                unions += uni.item()
        iou= (intersections + 1) / (unions + 1)
        dice_score = (up + 1) / (down + 1)
    model.train()
    return iou, dice_score, loss


#copied from module evaluate -> litte changes were needed
def evaluate(eval_loader, model, loss_fn, config, num_at_once=1, mode='stoic'):

    model.eval()

    all_preds = []
    all_inf_gt = []
    all_sev_gt = []
    all_patients = []

    running_loss = 0.
    running_loss_sev0 = 0.
    running_loss_sev1 = 0.

    with torch.no_grad():
        for sample in eval_loader:
            if mode == 'stoic':
                v_tensor, age, sex, inf_gt, sev_gt, patient = sample
            else:
                v_tensor, age, sex, inf_gt, sev_gt = sample
                patient = torch.ones((1,))


            all_inf_gt.append(inf_gt)
            all_sev_gt.append(sev_gt)
            all_patients.append(patient)

            loss_scaling = max((v_tensor.shape[0] // num_at_once), 1)

            preds = []
            for i in range(0, v_tensor.shape[0], num_at_once):
                step = min(i + num_at_once, v_tensor.shape[0])
                # select subset for gradient accumulation
                v_ten, s_gt, i_gt = v_tensor[i:step], sev_gt[i:step], inf_gt[i:step]
                v_ten, s_gt, i_gt = v_ten.to(config.DEVICE), s_gt.to(config.DEVICE), i_gt.to(config.DEVICE)
                # metadata
                if config.modelconfig.use_metadata and mode == 'stoic':
                    a = age[i:step].to(config.DEVICE)
                    s = sex[i:step].to(config.DEVICE)
                else:
                    a = None
                    s = None
                # forward pass
                pred_stoic, pred_mos, pred_hust, __ = model(v_ten, a, s, mode=mode)
                if mode == 'stoic':
                    pred = pred_stoic
                elif mode == 'mosmed':
                    pred = pred_mos
                else:
                    pred = pred_hust

                running_loss += loss_fn(pred, i_gt, s_gt).cpu().item() / loss_scaling

                # calculate extra loss for severe and non-severe cases
                l0, l1 = loss_fn.partial_loss(pred, i_gt, s_gt)
                running_loss_sev0 += l0.cpu().item() / loss_scaling
                running_loss_sev1 += l1.cpu().item() / loss_scaling

                preds.append(pred.cpu())

            all_preds.append(torch.cat(preds, dim=0))

    all_inf_gt = torch.cat(all_inf_gt)
    all_sev_gt = torch.cat(all_sev_gt)
    all_patients = torch.cat(all_patients)

    running_loss = running_loss / len(all_preds) # do this before torch.cat(all_preds)!
    running_loss_sev0 = running_loss_sev0 / len(all_preds)
    running_loss_sev1 = running_loss_sev1 / len(all_preds)
    all_preds = torch.cat(all_preds)

    pred_inf, pred_sev = loss_fn.finalize(all_preds)

    inf_roc = ev.rocauc_safe(all_inf_gt, pred_inf)
    sev_roc = ev.rocauc_safe(all_sev_gt, pred_sev)

    # for the submission, only patients with a COVID infection are counted
    covpats = (all_inf_gt == 1)
    submission_sev_roc = ev.rocauc_safe(all_sev_gt[covpats], pred_sev[covpats])

    model.train()

    return {
        "auc_inf": inf_roc,
        "auc_sev": sev_roc,
        "auc_sev2": submission_sev_roc,
        "loss": running_loss,
        "loss_sev0": running_loss_sev0,
        "loss_sev1": running_loss_sev1,
        "all_data": {
            "inf_gt": all_inf_gt,
            "inf_pred": pred_inf,
            "sev_gt": all_sev_gt,
            "sev_pred": pred_sev,
            "patients": all_patients,
        },
    }



def calc_stats():
    # hust
    args = parse_args()
    config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
    config.modelconfig = get_modelconfig(config)
    dataconfig = HustConfig(config)
    dataconfig.is_validation = True
    set_deterministic(1055)
    train_loader_mos = data.DataLoader(
        dataset=get_hust_dataset(dataconfig),
        num_workers=config.WORKERS,
        batch_size=1,
        shuffle=True,
        worker_init_fn=seed_worker
    )
    mean, var, num = 0, 0, 0
    for input, age, sex, inf_gt, sev_gt, __ in tqdm(train_loader_mos):
        mean += torch.mean(input)
        var += torch.var(input, unbiased=False)
        num += 1
        # print(num)
    mean, var = mean / num, var / num
    std = torch.sqrt(var)
    print('Hust mean:', mean)
    print('Hust var:', var)
    print('Hust std:', std)
    sys.stdout.flush()

    #mosmed
    args = parse_args()
    config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
    config.modelconfig = get_modelconfig(config)
    dataconfig = MosmedConfig(config)
    dataconfig.is_validation = True
    set_deterministic(1055)
    train_loader_mos = data.DataLoader(
        dataset=get_mosmed_dataset(dataconfig),
        num_workers=config.WORKERS,
        batch_size=1,
        shuffle=True,
        worker_init_fn=seed_worker
    )
    mean, var, num = 0, 0, 0
    for input, age, sex, inf_gt, sev_gt, __ in tqdm(train_loader_mos):
        mean += torch.mean(input)
        var += torch.var(input, unbiased=False)
        num += 1
        #print(num)
    mean, var = mean/num, var/num
    std = torch.sqrt(var)
    print('MosMed mean:', mean)
    print('MosMed var:', var)
    print('MosMed std:', std)
    sys.stdout.flush()

    #TCIA
    config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
    config.modelconfig = get_modelconfig(config)
    zoomsize = 224
    transform_seg_val = My_Compose([Zoom([zoomsize, zoomsize, zoomsize]), ToTensor3D()])
    train_dataset_seg = SupervisedDataset(path=config.DATA_PATH_SEG, transform=transform_seg_val, train=True)
    my_val_dataset_seg = SupervisedDataset(path=config.DATA_PATH_SEG, transform=transform_seg_val, train=False)
    train_loader_seg = torch.utils.data.DataLoader(train_dataset_seg, batch_size=1,
                                                   num_workers=config.WORKERS,
                                                   shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader_seg = torch.utils.data.DataLoader(my_val_dataset_seg, batch_size=1,
                                                    num_workers=config.WORKERS,
                                                    shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    mean, var, num = 0, 0, 0
    for loader in [train_loader_seg, my_val_loader_seg]:
        for input, label in tqdm(loader):
            mean += torch.mean(input)
            var += torch.var(input, unbiased=False)
            num += 1
            #print(num)
    mean, var = mean / num, var / num
    std = torch.sqrt(var)
    print('TCIA mean:', mean)
    print('TCIA var:', var)
    print('TCIA std:', std)
    sys.stdout.flush()

    #STOIC
    config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
    config.modelconfig = get_modelconfig(config)
    dataconfig = GPUDeformedDataConfig(config)
    dataconfig.is_validation = True
    dataset = get_dataset(dataconfig)
    stoic_loader = data.DataLoader(
        dataset=dataset, num_workers=config.WORKERS, batch_size=1,
        shuffle=True,
        worker_init_fn=seed_worker
    )
    mean, var, num = 0, 0, 0
    for input, age, sex, inf_gt, sev_gt, __ in tqdm(stoic_loader):
        mean += torch.mean(input)
        var += torch.var(input, unbiased=False)
        num += 1
        #print(num)
    mean, var = mean / num, var / num
    std = torch.sqrt(var)
    print('STOIC mean:', mean)
    print('STOIC var:', var)
    print('STOIC std:', std)
    sys.stdout.flush()





def main():
    print('PID:', os.getpid())
    set_deterministic()
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    logging.getLogger().setLevel(logging.INFO)

    config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
    config.modelconfig = get_modelconfig(config)
    dtime = config.datetime

    if args.cv:
        for fold in range(config.num_folds):
            print('-----------')
            print('New fold:', fold)
            print('------------')
            config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
            config.modelconfig = get_modelconfig(config)
            config.datetime = dtime
            #config.datetime = '2022-04-05_14-34-25'
            set_deterministic(1055)
            config.set_fold(fold)
            run(config, args)
    else:
        set_deterministic(1055)
        config.set_fold(0)
        run(config, args)


def start_experiment(config, args):
    print('PID:', os.getpid())
    set_deterministic()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    logging.getLogger().setLevel(logging.INFO)

    dtime = config.datetime

    if args.cv:
        for fold in range(config.num_folds):
            print('-----------')
            print('New fold:', fold)
            print('------------')
            #config = MultiConfig(args, cv_enabled=args.cv, split="cv5_infonly_bal.csv")
            #config.modelconfig = get_modelconfig(config)
            #config.datetime = dtime
            # config.datetime = '2022-04-05_14-34-25'
            set_deterministic(1055)
            config.set_fold(fold)
            run(config, args)
    else:
        set_deterministic(1055)
        config.set_fold(0)
        run(config, args)


if __name__ == '__main__':
    main()
    #calc_stats()

