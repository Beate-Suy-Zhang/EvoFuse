import os
from pathlib import Path

import yaml


HYPERPARAMETER_FILE = Path(__file__).resolve().with_name("main_hyperparameters.yaml")


def load_hyperparameters(config_path=HYPERPARAMETER_FILE):
    """Load and minimally validate the experiment hyperparameters."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Hyperparameter file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Hyperparameter file must contain a YAML mapping: {config_path}")

    required_sections = {
        "experiment",
        "evolutionary_learning",
        "image_fusion",
        "multi_object_detection",
        "semantic_segmentation",
        "salient_object_detection",
    }
    missing_sections = sorted(required_sections.difference(config))
    if missing_sections:
        raise KeyError(f"Missing hyperparameter sections: {', '.join(missing_sections)}")

    return config


HYPERPARAMETERS = load_hyperparameters()
EXPERIMENT_HYP = HYPERPARAMETERS["experiment"]
EVOLUTION_HYP = HYPERPARAMETERS["evolutionary_learning"]
FUSION_HYP = HYPERPARAMETERS["image_fusion"]
DETECTION_HYP = HYPERPARAMETERS["multi_object_detection"]
SEGMENTATION_HYP = HYPERPARAMETERS["semantic_segmentation"]
SOD_HYP = HYPERPARAMETERS["salient_object_detection"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


if __name__ == '__main__':
    print(os.getcwd())
    print(f"Loaded hyperparameters from {HYPERPARAMETER_FILE}")
################################## GPU ID ##################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from ultralytics import YOLO

from test_seg import test_seg
from train_seg import train_seg
from test_sod import test_sod
from train_sod import train_sod
import glob

import sys
import time
import datetime
import kornia
from tqdm import tqdm
import pygad
import numpy as np
import warnings
import logging
import copy

import cv2
from PIL import Image


################################## PATH CONFIGURATION ##################################
from evo_path_config.paths import get_path_config
from thop import profile

################################## TOOLS ##################################
from futools.netdraevo import Encoder, Decoder, FusionModel, prominent, DRA1BR
from futools.utils.dataset import H5Dataset
from futools.utils.loss import Fusionloss, pmt_loss, vsm_loss
from futools.utils.img_read_save import img_save
from futools.utils.testloader import get_test_loader
from detutils.dataloaders import create_dataloader

from segutils.segdataloader import SegSegDataset
from segutils.model import WeTr
from segutils.optimizer import PolyWarmupAdamW
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss

from detutils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_img_size,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from detutils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP,
                               smart_optimizer, smart_resume, torch_distributed_zero_first)
from detutils.loss_tal import ComputeLoss
from models.yolo import Model

from sodutils.dataloader.siam_fus_sod_dataloader import get_sod_loader
from sodutils.models.fgccnet import FGCCNet
from sodutils.loss import sodloss
from sodutils.utility import clip_gradient

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the device used
TQDM_BAR_FORMAT = '{l_bar}{bar:50}| {n_fmt}/{total_fmt} {elapsed}'
TQDM_BAR_FORMAT_2 = '{l_bar}{bar:30}| {n_fmt}/{total_fmt} {elapsed}'

# Initial Fusion Config 
criteria_fusion = Fusionloss()
MSELoss = nn.MSELoss()
Loss_ssim = kornia.losses.SSIMLoss(FUSION_HYP["ssim_window_size"], reduction='mean')


ckpt_path = EXPERIMENT_HYP["initial_fusion_checkpoint"]


Fusion_Model = FusionModel().to(device)
Fusion_Model.load_state_dict(torch.load(ckpt_path)['Fusion_Model'])
best_Model = copy.deepcopy(Fusion_Model).to(device)

# train_data_path = FUSION_HYP["h5_training"]["data_path"]
trainloader = None

evodata = list(EXPERIMENT_HYP["datasets"])
(project_root, train_folder_list, train_irfolder_list, train_vifolder_list, train_loader_list, val_folder_list, 
val_irfolder_list, val_vifolder_list, val_loader_list, project_root_list) = get_path_config(evodata)

best_fit = 0
best_fits = [0]*len(evodata)
pop_num = 0
iter_name = 'iter0'

previous_solution = []
current_epoch = 0

# Genetic algorithm parameters loaded from YAML
numberGeneration = EVOLUTION_HYP["num_generations"]
numberParentsMating = EVOLUTION_HYP["num_parents_mating"]
solutionPerPopulation = EVOLUTION_HYP["solutions_per_population"]
parents = EVOLUTION_HYP["keep_parents"]
geneType = {"float": float, "int": int}[EVOLUTION_HYP["gene_type"]]
minValue = EVOLUTION_HYP["init_range_low"]
maxValue = EVOLUTION_HYP["init_range_high"]
selectionType = EVOLUTION_HYP["parent_selection_type"]
crossoverType = EVOLUTION_HYP["crossover_type"]
crossoverRate = EVOLUTION_HYP["crossover_probability"]
mutationType = EVOLUTION_HYP["mutation_type"]
mutationReplacement = EVOLUTION_HYP["mutation_by_replacement"]
mutationRate = EVOLUTION_HYP["mutation_percent_genes"]


def EvolutionSODFusion(iter_num=0, train_task_index=0):
    global Fusion_Model, best_fit, pop_num, best_Model, device, \
            project_root, current_epoch, best_fits
    '''
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    '''
    # cudnn 在运行时自动寻找最适合当前配置的卷积网络
    torch.backends.cudnn.benchmark = True
    num_epochs = EVOLUTION_HYP["epochs_per_task"]
    lr0 = EVOLUTION_HYP["initial_learning_rate"]
    lrs = [lr0 * (num_epochs - i + 1) / num_epochs for i in range(1, 1 + num_epochs)]
    numberGenes = EVOLUTION_HYP["num_genes"]["salient_object_detection"]

    for epoch in range(num_epochs):
        print("\n"*2+"="*80)
        print("Epoch: ", epoch + 1, " / ", num_epochs)
        current_epoch = epoch

        def fitnessFunction(geneticAlgorithm, solution, solution_idx):
            global Fusion_Model, best_fit, pop_num, best_Model, iter_name, device,\
                project_root, current_epoch, best_fits
            print()
            print()
            print("Genetic total number: ", pop_num, "      Individual index: ", solution_idx, "      lr: ", lrs[current_epoch])
            pop_num = pop_num + 1
            print("solution:\t", solution)
            
            copy_Model = copy.deepcopy(Fusion_Model).to(device)
            ############################################################################################
            ############################################################################################
            ############################################################################################
            train_fusod_one_epoch(train_loader_list[train_task_index], 
                                train_task_index, copy_Model, lrs[current_epoch], 
                                solution
            )
            acc, accs = validationModelOnTasks(copy_Model, val_loader_list, val_folder_list, project_root_list, iter_name)
            print()
            if acc > best_fit:
                best_fit = acc
                best_fits = accs
                best_Model = copy.deepcopy(copy_Model).to(device)
                checkpoint = {
                    'Fusion_Model': best_Model.state_dict(),
                }
                fusion_model_output_path = os.path.join(project_root, "models")
                if not os.path.exists(fusion_model_output_path):
                    os.makedirs(fusion_model_output_path)
                torch.save(checkpoint, os.path.join(fusion_model_output_path, "evovvtfuse_"+str(acc)[0:8]+'.pth'))
                
            
            return acc

        geneticAlgorithm = pygad.GA(
            num_generations=numberGeneration,
            num_parents_mating=numberParentsMating,
            num_genes=numberGenes,
            gene_type=geneType,
            fitness_func=fitnessFunction,
            sol_per_pop=solutionPerPopulation,
            init_range_high=maxValue,   
            init_range_low=minValue,
            parent_selection_type=selectionType,
            keep_parents=parents,
            crossover_type=crossoverType,
            crossover_probability=crossoverRate,
            mutation_type=mutationType,
            mutation_by_replacement=mutationReplacement,
            random_mutation_max_val=EVOLUTION_HYP["random_mutation_max_val"],
            random_mutation_min_val=EVOLUTION_HYP["random_mutation_min_val"],
            mutation_percent_genes=mutationRate,
            save_solutions=EVOLUTION_HYP["save_solutions"],
            save_best_solutions=EVOLUTION_HYP["save_best_solutions"],
            suppress_warnings=EVOLUTION_HYP["suppress_warnings"],
        )
        
        pop_num = 0
        org_pop = copy.deepcopy(geneticAlgorithm.population)
        previous_solution = org_pop
        
        if epoch > 0:
            geneticAlgorithm.population = copy.deepcopy(
                previous_solution * EVOLUTION_HYP["previous_population_weight"]
                + geneticAlgorithm.population * EVOLUTION_HYP["random_population_weight"]
            )
            
        geneticAlgorithm.run()
        best_solution, best_solution_fitness, best_solution_idx = geneticAlgorithm.best_solution()
        Fusion_Model = copy.deepcopy(best_Model).to(device)
        print()
        print()
        print('original solution: ')
        print(org_pop)
        print('Evolutionary solution: ')
        print(geneticAlgorithm.population)
        print('best solution: ', best_solution)
        print('best fitness: ', best_fit)
            
        if True:
            checkpoint = {
                'Fusion_Model': best_Model.state_dict(),
            }
            fusion_model_output_path = os.path.join(project_root, "models")
            if not os.path.exists(fusion_model_output_path):
                os.makedirs(fusion_model_output_path)
            torch.save(checkpoint, os.path.join(fusion_model_output_path, "iter"+str(iter_num)+
                                                "_evoepoch"+str(epoch+1)+"_tinyfuse_"+str(best_fit)[0:6]+'.pth'))
        print()
    return os.path.join(project_root, "evos/iter"+str(iter_num)+"_evoepoch"+str(epoch+1)+"_tinyfuse_"
                                                +str(best_fit)[0:6])


def EvolutionDetectionFusion(iter_num=0, train_task_index=0):
    global Fusion_Model, best_fit, pop_num, best_Model, device, \
            project_root, current_epoch, best_fits
    '''
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    '''
    # cudnn 在运行时自动寻找最适合当前配置的卷积网络
    torch.backends.cudnn.benchmark = True
    # Set the Hyper-parameters for Evolutionary Steps
    num_epochs = EVOLUTION_HYP["epochs_per_task"]
    lr0 = EVOLUTION_HYP["initial_learning_rate"]
    lrs = [lr0 * (num_epochs - i + 1) / num_epochs for i in range(1, 1 + num_epochs)]
    numberGenes = EVOLUTION_HYP["num_genes"]["multi_object_detection"]

    for epoch in range(num_epochs):
        print("\n"*2+"="*80)
        print("Epoch: ", epoch + 1, " / ", num_epochs)
        current_epoch = epoch

        def fitnessFunction(geneticAlgorithm, solution, solution_idx):
            global Fusion_Model, best_fit, pop_num, best_Model, iter_name, device,\
                project_root, current_epoch, best_fits
            print()
            print()
            print("Genetic total number: ", pop_num, "      Individual index: ", solution_idx, "      lr: ", lrs[current_epoch])
            pop_num = pop_num + 1
            print("solution:\t", solution)
            
            # copy best fusion model
            copy_Model = copy.deepcopy(Fusion_Model).to(device)
            # randomly initialize the optimizers
            copy_optimizer1 = torch.optim.Adam(
                copy_Model.parameters(),
                lr=lrs[current_epoch],
                weight_decay=FUSION_HYP["optimizer"]["weight_decay"],
            )
            ############################################################################################
            ############################################################################################
            ############################################################################################
            train_fudet_one_epoch(train_loader_list[train_task_index], 
                                train_task_index, copy_Model, lrs[current_epoch], 
                                solution
                            )
            acc, accs = validationModelOnTasks(copy_Model, val_loader_list, val_folder_list, project_root_list, iter_name)
            print()
            if acc > best_fit:
                best_fit = acc
                best_fits = accs
                best_Model = copy.deepcopy(copy_Model).to(device)
                checkpoint = {
                    'Fusion_Model': best_Model.state_dict(),
                }
                fusion_model_output_path = os.path.join(project_root, "models")
                if not os.path.exists(fusion_model_output_path):
                    os.makedirs(fusion_model_output_path)
                torch.save(checkpoint, os.path.join(fusion_model_output_path, "evovvtfuse_"+str(acc)[0:8]+'.pth'))
                
            
            return acc

        geneticAlgorithm = pygad.GA(
            num_generations=numberGeneration,
            num_parents_mating=numberParentsMating,
            num_genes=numberGenes,
            gene_type=geneType,
            fitness_func=fitnessFunction,
            sol_per_pop=solutionPerPopulation,
            init_range_high=maxValue,   
            init_range_low=minValue,
            parent_selection_type=selectionType,
            keep_parents=parents,
            crossover_type=crossoverType,
            crossover_probability=crossoverRate,
            mutation_type=mutationType,
            mutation_by_replacement=mutationReplacement,
            random_mutation_max_val=EVOLUTION_HYP["random_mutation_max_val"],
            random_mutation_min_val=EVOLUTION_HYP["random_mutation_min_val"],
            mutation_percent_genes=mutationRate,
            save_solutions=EVOLUTION_HYP["save_solutions"],
            save_best_solutions=EVOLUTION_HYP["save_best_solutions"],
            suppress_warnings=EVOLUTION_HYP["suppress_warnings"],
        )
        
        pop_num = 0
        org_pop = copy.deepcopy(geneticAlgorithm.population)
        previous_solution = org_pop
        
        if epoch > 0:
            geneticAlgorithm.population = copy.deepcopy(
                previous_solution * EVOLUTION_HYP["previous_population_weight"]
                + geneticAlgorithm.population * EVOLUTION_HYP["random_population_weight"]
            )
            
        geneticAlgorithm.run()
        best_solution, best_solution_fitness, best_solution_idx = geneticAlgorithm.best_solution()
        Fusion_Model = copy.deepcopy(best_Model).to(device)
        print()
        print()
        print('original solution: ')
        print(org_pop)
        print('Evolutionary solution: ')
        print(geneticAlgorithm.population)
        print('best solution: ', best_solution)
        print('best fitness: ', best_fit)
            
        if True:
            checkpoint = {
                'Fusion_Model': best_Model.state_dict(),
            }
            fusion_model_output_path = os.path.join(project_root, "models")
            if not os.path.exists(fusion_model_output_path):
                os.makedirs(fusion_model_output_path)
            torch.save(checkpoint, os.path.join(fusion_model_output_path, "iter"+str(iter_num)+
                                                "_evoepoch"+str(epoch+1)+"_tinyfuse_"+str(best_fit)[0:6]+'.pth'))
        print()
    return os.path.join(project_root, "evos/iter"+str(iter_num)+"_evoepoch"+str(epoch+1)+"_tinyfuse_"
                                                +str(best_fit)[0:6])
    
    
def EvolutionSegmentationFusion(iter_num=0, train_task_index=1):
    global Fusion_Model, best_fit, pop_num, best_Model, device, \
            project_root, current_epoch, best_fits
    '''
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    '''
    # cudnn 在运行时自动寻找最适合当前配置的卷积网络
    torch.backends.cudnn.benchmark = True
    # Set the Hyper-parameters for Evolutionary Steps
    num_epochs = EVOLUTION_HYP["epochs_per_task"]
    lr0 = EVOLUTION_HYP["initial_learning_rate"]
    lrs = [lr0 * (num_epochs - i + 1) / num_epochs for i in range(1, 1 + num_epochs)]
    numberGenes = EVOLUTION_HYP["num_genes"]["semantic_segmentation"]

    for epoch in range(num_epochs):
        print("\n"*2+"="*80)
        print("Epoch: ", epoch + 1, " / ", num_epochs)
        current_epoch = epoch

        def fitnessFunction(geneticAlgorithm, solution, solution_idx):
            global Fusion_Model, best_fit, pop_num, best_Model, iter_name, device,\
                project_root, current_epoch, best_fits
            print()
            print()
            print("Genetic total number: ", pop_num, "      Individual index: ", solution_idx, "      lr: ", lrs[current_epoch])
            pop_num = pop_num + 1
            print("solution:\t", solution)
            
            # copy best fusion model
            copy_Model = copy.deepcopy(Fusion_Model).to(device)
            ############################################################################################
            ############################################################################################
            ############################################################################################
            train_fuseg_one_epoch(train_loader_list[train_task_index], 
                                train_task_index, copy_Model, lrs[current_epoch], 
                                solution
                            )
            acc, accs = validationModelOnTasks(copy_Model, val_loader_list, val_folder_list, project_root_list, iter_name)
            print()
            if acc > best_fit:
                best_fit = acc
                best_fits = accs
                best_Model = copy.deepcopy(copy_Model).to(device)
                checkpoint = {
                    'Fusion_Model': best_Model.state_dict(),
                }
                fusion_model_output_path = os.path.join(project_root, "models")
                if not os.path.exists(fusion_model_output_path):
                    os.makedirs(fusion_model_output_path)
                torch.save(checkpoint, os.path.join(fusion_model_output_path, "evovvtfuse_"+str(acc)[0:8]+'.pth'))
                
            
            return acc

        geneticAlgorithm = pygad.GA(
            num_generations=numberGeneration,
            num_parents_mating=numberParentsMating,
            num_genes=numberGenes,
            gene_type=geneType,
            fitness_func=fitnessFunction,
            sol_per_pop=solutionPerPopulation,
            init_range_high=maxValue,   
            init_range_low=minValue,
            parent_selection_type=selectionType,
            keep_parents=parents,
            crossover_type=crossoverType,
            crossover_probability=crossoverRate,
            mutation_type=mutationType,
            mutation_by_replacement=mutationReplacement,
            random_mutation_max_val=EVOLUTION_HYP["random_mutation_max_val"],
            random_mutation_min_val=EVOLUTION_HYP["random_mutation_min_val"],
            mutation_percent_genes=mutationRate,
            save_solutions=EVOLUTION_HYP["save_solutions"],
            save_best_solutions=EVOLUTION_HYP["save_best_solutions"],
            suppress_warnings=EVOLUTION_HYP["suppress_warnings"],
        )
        
        pop_num = 0
        org_pop = copy.deepcopy(geneticAlgorithm.population)
        previous_solution = org_pop
        
        if epoch > 0:
            geneticAlgorithm.population = copy.deepcopy(
                previous_solution * EVOLUTION_HYP["previous_population_weight"]
                + geneticAlgorithm.population * EVOLUTION_HYP["random_population_weight"]
            )
            
        geneticAlgorithm.run()
        best_solution, best_solution_fitness, best_solution_idx = geneticAlgorithm.best_solution()
        Fusion_Model = copy.deepcopy(best_Model).to(device)
        print()
        print()
        print('original solution: ')
        print(org_pop)
        print('Evolutionary solution: ')
        print(geneticAlgorithm.population)
        print('best solution: ', best_solution)
        print('best fitness: ', best_fit)
            
        if True:
            checkpoint = {
                'Fusion_Model': best_Model.state_dict(),
            }
            fusion_model_output_path = os.path.join(project_root, "models")
            if not os.path.exists(fusion_model_output_path):
                os.makedirs(fusion_model_output_path)
            torch.save(checkpoint, os.path.join(fusion_model_output_path, "iter"+str(iter_num)+
                                                "_evoepoch"+str(epoch+1)+"_tinyfuse_"+str(best_fit)[0:6]+'.pth'))
        print()
    return os.path.join(project_root, "evos/iter"+str(iter_num)+"_evoepoch"+str(epoch+1)+"_tinyfuse_"
                                                +str(best_fit)[0:6])
            

# def train_one_epoch(train_loader, Fusion_Model, optimizer1,  
#                     coeff_pmt_IF, coeff_pmt_VF, coeff_int, coeff_grad, coeff_mse_I, coeff_mse_V,
#                     ):
#     ''' train '''
#     prev_time = time.time()
#     global trainloader
#     h5_hyp = FUSION_HYP["h5_training"]
#     trainloader = DataLoader(
#         H5Dataset(train_data_path),
#         batch_size=h5_hyp["batch_size"],
#         shuffle=h5_hyp["shuffle"],
#         num_workers=h5_hyp["num_workers"],
#     )

#     for i, (data_VIS, data_IR) in enumerate(tqdm(trainloader, bar_format=TQDM_BAR_FORMAT)):
#         if i % h5_hyp["train_every_n_batches"] == 0:
            
#             data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
#             Fusion_Model.train()
#             Fusion_Model.zero_grad()
#             optimizer1.zero_grad()

#             data_Fuse = Fusion_Model(data_VIS, data_IR)
            
#             mse_loss_V = (Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)) * coeff_mse_V
#             mse_loss_I = (Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)) * coeff_mse_I
#             pmt_loss_I_F = pmt_loss(data_IR, data_Fuse) * coeff_pmt_IF
#             pmt_loss_V_F = pmt_loss(data_VIS, data_Fuse) * coeff_pmt_VF
#             totalfusionloss, fusionloss, loss_grad = criteria_fusion(data_VIS, data_IR, data_Fuse)
#             loss =   fusionloss * coeff_int + loss_grad * coeff_grad + pmt_loss_I_F + pmt_loss_V_F  + \
#                     mse_loss_V + mse_loss_I
                    
#             loss.backward()

#             nn.utils.clip_grad_norm_(
#                 Fusion_Model.parameters(),
#                 max_norm=FUSION_HYP["gradient_clipping"]["max_norm"],
#                 norm_type=FUSION_HYP["gradient_clipping"]["norm_type"],
#             )
#             optimizer1.step()
#     del trainloader
#     print("epochfitnessDone")
#     print()
            

def train_fuseg_one_epoch(train_loader, train_task_index, Fusion_Model, lr,  
                    solution,
                    ):
    ''' train '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    prev_time = time.time()
    global trainloader, iter_name, project_root_list, evodata, device
    
    seg_hyp = SEGMENTATION_HYP
    dataset_hyp = seg_hyp["dataset"]
    dataloader_hyp = seg_hyp["dataloader"]
    optimizer_hyp = seg_hyp["optimizer"]
    dataset_name = evodata[train_task_index]
    num_classes = dataset_hyp["num_classes_by_dataset"].get(
        dataset_name, dataset_hyp["default_num_classes"]
    )
    batch_size = seg_hyp["evolution_batch_size"]
    trainloader = get_test_loader(train_loader[0], train_loader[1], train_loader[2], train_loader[3], batch_size)
    
    seg_train_dataset = SegSegDataset(
        root_dir=os.path.join('./datasets', dataset_name),
        name_list_dir=None,
        split=dataset_hyp["split"],
        stage=dataset_hyp["stage"],
        aug=dataset_hyp["aug"],
        resize_range=dataset_hyp["resize_range"],
        rescale_range=dataset_hyp["rescale_range"],
        crop_size=dataset_hyp["crop_size"],
        img_fliplr=dataset_hyp["horizontal_flip"],
        ignore_index=dataset_hyp["ignore_index"],
        num_classes=num_classes,
    )
    seg_train_loader = DataLoader(seg_train_dataset,
                              batch_size=batch_size,
                              shuffle=dataloader_hyp["shuffle"],
                              num_workers=dataloader_hyp["num_workers"],
                              pin_memory=dataloader_hyp["pin_memory"],
                              drop_last=dataloader_hyp["drop_last"],
                            #   sampler=train_sampler,
                              prefetch_factor=dataloader_hyp["prefetch_factor"])
    
    segmodel = WeTr(backbone=seg_hyp["model"]["backbone"],
                num_classes=num_classes,
                embedding_dim=seg_hyp["model"]["embedding_dim"],
                pretrained=seg_hyp["model"]["pretrained"],
                pretrained_path=glob.glob(project_root_list[train_task_index] + iter_name + "/best_mIoU*.pth")[0],).to(device)
    param_groups = segmodel.get_param_groups()
    
    lr = lr
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": Fusion_Model.parameters(),
                "lr": lr,
                "weight_decay": optimizer_hyp["fusion_weight_decay"],
            },
            {
                "params": param_groups[0],
                "lr": lr,
                "weight_decay": optimizer_hyp["backbone_group_0_weight_decay"],
            },
            {
                "params": param_groups[1],
                "lr": lr,
                "weight_decay": optimizer_hyp["backbone_group_1_weight_decay"],
            },
            {
                "params": param_groups[2],
                "lr": lr * optimizer_hyp["backbone_group_2_lr_multiplier"],
                "weight_decay": optimizer_hyp["backbone_group_2_weight_decay"],
            },
        ],
        lr = lr,
        weight_decay = optimizer_hyp["weight_decay"],
        betas = optimizer_hyp["betas"],
        warmup_iter = optimizer_hyp["warmup_iterations"],
        max_iter = optimizer_hyp["max_iterations"],
        warmup_ratio = optimizer_hyp["warmup_ratio"],
        power = optimizer_hyp["power"],
    )
    
    segLoss = CrossEntropyLoss()
    segLoss = segLoss.to(device)
    
    pbar = (seg_train_loader)
    pbar = tqdm(pbar, bar_format=TQDM_BAR_FORMAT, leave=False)
    
    for i, ((_, _, labels), (irimage, visimage_rgb, image_name))  \
        in enumerate(zip(pbar, trainloader)):
    
        if i >= len(pbar) / 10:
            break

        if i % seg_hyp["train_every_n_batches"] == 0:
            segmodel.train()
            segmodel.zero_grad()
            Fusion_Model.train()
            Fusion_Model.zero_grad()
            optimizer.zero_grad()
            
            data_IR = irimage.to(device)
            data_Ycbcr = visimage_rgb.to(device)
            data_VIS = data_Ycbcr[:, 0:1, :, :]
            labels = labels.to(device, non_blocking=True)

            data_Fuse = Fusion_Model(data_VIS, data_IR)
            
            pmt_loss_I_F = pmt_loss(data_IR, data_Fuse) * solution[0]
            pmt_loss_V_F = pmt_loss(data_VIS, data_Fuse) * solution[1]
            mse_loss_V = (Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)) * solution[2]
            mse_loss_I = (Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)) * solution[3]
            totalfusionloss, fusionloss, loss_grad = criteria_fusion(data_VIS, data_IR, data_Fuse)
            fusionloss = fusionloss * solution[4]
            loss_grad = loss_grad * solution[5]
            fuloss =   fusionloss + loss_grad + pmt_loss_I_F + pmt_loss_V_F + mse_loss_V + mse_loss_I
                    
            Y = data_Fuse
            Cb = data_Ycbcr[:, 1:2, :, :]
            Cr = data_Ycbcr[ :, 2:3, :, :]
            R = Y + 1.402 * (Cr - 0.5)
            G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
            B = Y + 1.772 * (Cb - 0.5)
            # data_RGB = torch.cat((B, G, R), dim=1)
            data_RGB = torch.cat((B, G, R), 1)
            data_RGB = torch.clamp(data_RGB, 0., 1.)  # 确保值范围在 [0, 1]
            imgs = data_RGB
                    
            pred = segmodel(imgs)  # forward
            pred = F.interpolate(pred, size=labels.shape[1:], mode='bilinear', align_corners=False)
            
            seg_loss = segLoss(pred, labels.type(torch.long)) * solution[6]
            
            
            loss = seg_loss + fuloss
            loss.backward()

            nn.utils.clip_grad_norm_(
                Fusion_Model.parameters(),
                max_norm=FUSION_HYP["gradient_clipping"]["max_norm"],
                norm_type=FUSION_HYP["gradient_clipping"]["norm_type"],
            )
            optimizer.step()
    del trainloader, seg_train_dataset, segmodel, segLoss, pbar, optimizer, param_groups, seg_train_loader

    print("epochfitnessDone")
    print()  
    

def train_fudet_one_epoch(train_loader, train_task_index, Fusion_Model, lr,  
                    solution,
                    ):
    ''' train '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    prev_time = time.time()
    global trainloader, iter_name, project_root_list, evodata
    
    det_hyp = DETECTION_HYP
    dataloader_hyp = det_hyp["dataloader"]
    batch_size = det_hyp["evolution_batch_size"]
    # VIIR loader for image fusion
    trainloader = get_test_loader(train_loader[0], train_loader[1], train_loader[2], train_loader[3], batch_size)    
    
    # Trainloader
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    
    data = os.path.join(det_hyp["dataset_yaml_directory"], evodata[train_task_index] + ".yaml")
    hyp = det_hyp["hyperparameter_file"]
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(data)  # check if None
    imgsz = train_loader[3][0]
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # number of classes
    
    cfg = det_hyp["model_config"]
    weights = project_root_list[train_task_index] + iter_name + "/weights/best.pt"
    ckpt = torch.load(weights, map_location=det_hyp["checkpoint_map_location"])
    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), verbose=False).to(device)  # create
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    
    compute_loss = ComputeLoss(model)  # init loss class
    
    gs = max(int(model.stride.max()), det_hyp["minimum_grid_size"])
    single_cls = det_hyp["single_class"]
    det_train_loader, dataset = create_dataloader(train_path,
                                                imgsz,
                                                batch_size,
                                                gs,
                                                single_cls,
                                                hyp=hyp,
                                                augment=dataloader_hyp["augment"],
                                                cache=dataloader_hyp["cache"],
                                                rect=dataloader_hyp["rect"],
                                                rank=LOCAL_RANK,
                                                workers=dataloader_hyp["workers"],
                                                image_weights=dataloader_hyp["image_weights"],
                                                close_mosaic=dataloader_hyp["close_mosaic"],
                                                quad=dataloader_hyp["quad"],
                                                prefix=colorstr('train: '),
                                                shuffle=dataloader_hyp["shuffle"],
                                                min_items=dataloader_hyp["min_items"],)
    pbar = (det_train_loader)
    pbar = tqdm(pbar, bar_format=TQDM_BAR_FORMAT, leave=False)
    
    
    optimizer1 = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': Fusion_Model.parameters()}
        ], lr=lr, weight_decay=det_hyp["optimizer"]["weight_decay"])
    
    for i, ((imgs, targets, paths, _), (irimage, visimage_rgb, image_name))  \
        in enumerate((zip(pbar, trainloader))):
        if i >= len(pbar) / 10:
            break

        if i % det_hyp["train_every_n_batches"] == 0:
            model.train()
            model.zero_grad()
            Fusion_Model.train()
            Fusion_Model.zero_grad()
            optimizer1.zero_grad()
            
            data_IR = irimage.to(device)
            data_Ycbcr = visimage_rgb.to(device)
            data_VIS = data_Ycbcr[:, 0:1, :, :]

            data_Fuse = Fusion_Model(data_VIS, data_IR)
            
            pmt_loss_I_F = pmt_loss(data_IR, data_Fuse) * solution[0]
            pmt_loss_V_F = pmt_loss(data_VIS, data_Fuse) * solution[1]
            mse_loss_V = (Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)) * solution[2]
            mse_loss_I = (Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)) * solution[3]
            totalfusionloss, fusionloss, loss_grad = criteria_fusion(data_VIS, data_IR, data_Fuse)
            fusionloss = fusionloss * solution[4]
            loss_grad = loss_grad * solution[5]
            fuloss =   fusionloss + loss_grad + pmt_loss_I_F + pmt_loss_V_F + mse_loss_V + mse_loss_I
                    
            Y = data_Fuse
            Cb = data_Ycbcr[:, 1:2, :, :]
            Cr = data_Ycbcr[ :, 2:3, :, :]
            R = Y + 1.402 * (Cr - 0.5)
            G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
            B = Y + 1.772 * (Cb - 0.5)
            # data_RGB = torch.cat((B, G, R), dim=1)
            data_RGB = torch.cat((B, G, R), 1)
            data_RGB = torch.clamp(data_RGB, 0., 1.)  # 确保值范围在 [0, 1]
            imgs = data_RGB
            fu_padding = tuple(det_hyp["fused_image_padding"])
            imgs = torch.nn.functional.pad(
                imgs,
                fu_padding,
                mode='constant',
                value=det_hyp["fused_image_padding_value"],
            )
                    
            # with torch.cuda.amp.autocast(amp):
            pred = model(imgs)  # forward
            boxg = solution[6]
            clsg = solution[7]
            dflg = solution[8]
            det_loss, li = compute_loss(pred, targets.to(device), boxgain=boxg, clsgain=clsg, dflgain=dflg)
            # print('det_loss', li)
            
            loss = det_loss + fuloss
            loss.backward()

            nn.utils.clip_grad_norm_(
                Fusion_Model.parameters(),
                max_norm=FUSION_HYP["gradient_clipping"]["max_norm"],
                norm_type=FUSION_HYP["gradient_clipping"]["norm_type"],
            )
            optimizer1.step()
    del (trainloader, det_train_loader, model, dataset, ckpt, pbar, hyp, data_dict, csd, compute_loss, optimizer1, 
            train_path, val_path, weights, )

    print("epochfitnessDone")
    print()
    

def train_fusod_one_epoch(train_loader, train_task_index, Fusion_Model, lr,  solution,):
    ''' train '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    prev_time = time.time()
    global trainloader, iter_name, project_root_list, evodata
    
    sod_hyp = SOD_HYP
    train_path = "./datasets/" + evodata[train_task_index]
    # rgb_root     = train_path + '/train/RGB/'
    # thermal_root = train_path + '/train/T/'
    gt_root      = train_path + '/train/GT/'

    batchsize = sod_hyp["evolution_batch_size"]
    trainsize = sod_hyp["train_size"]
    # trainloader = get_test_loader(train_loader[0], train_loader[1], train_loader[2], train_loader[3], batch_size) 
    train_loader = get_sod_loader(train_loader[1], train_loader[0], gt_root, # opt.rgb_root  opt.thermal_root
                                batchsize=batchsize,
                                trainsize=trainsize,
                                shuffle=sod_hyp["shuffle"],
                                split=sod_hyp["split"]
                                )
    sodnet    = FGCCNet().to(device)
    
    optimizer = torch.optim.Adam([
            {'params': sodnet.parameters()},
            {'params': Fusion_Model.parameters()}
        ], lr=lr, weight_decay=sod_hyp["optimizer"]["weight_decay"])
    
    criteria_sod = sodloss.SODLoss().to(device)
    
    load_path = os.path.join(project_root_list[train_task_index], iter_name, "model_sod.pth")
    sodnet.load_state_dict(torch.load(load_path))
    print('===> Loading pretrained model from {} sucessfully~')
    
    for i, (visimage_rgb, visimage_ycbcr, irimage, gt, image_name) in tqdm(enumerate(train_loader), bar_format=TQDM_BAR_FORMAT):
        
        if i >= len(train_loader) / 10:
            break

        if i % sod_hyp["train_every_n_batches"] == 0:
            
            sodnet.train()
            sodnet.zero_grad()
            Fusion_Model.train()
            Fusion_Model.zero_grad()
            optimizer.zero_grad()

            rgb = visimage_rgb.cuda()
            thermal = irimage.cuda()
            # fus = fus.cuda()
            gt = gt.cuda()
            
            data_IR = irimage.to(device)
            gray_weights = sod_hyp["infrared_grayscale_weights"]
            data_IR = (
                gray_weights[0] * data_IR[:, 0:1, :, :]
                + gray_weights[1] * data_IR[:, 1:2, :, :]
                + gray_weights[2] * data_IR[:, 2:3, :, :]
            )
            data_Ycbcr = visimage_ycbcr.to(device)
            data_VIS = data_Ycbcr[:, 0:1, :, :]

            data_Fuse = Fusion_Model(data_VIS, data_IR)

            data_Fuse = Fusion_Model(data_VIS, data_IR)
            
            pmt_loss_I_F = pmt_loss(data_IR, data_Fuse) * solution[0]
            pmt_loss_V_F = pmt_loss(data_VIS, data_Fuse) * solution[1]
            mse_loss_V = (Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)) * solution[2]
            mse_loss_I = (Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)) * solution[3]
            totalfusionloss, fusionloss, loss_grad = criteria_fusion(data_VIS, data_IR, data_Fuse)
            fusionloss = fusionloss * solution[4]
            loss_grad = loss_grad * solution[5]
            fuloss =   fusionloss + loss_grad + pmt_loss_I_F + pmt_loss_V_F + mse_loss_V + mse_loss_I
                    
            Y = data_Fuse
            Cb = data_Ycbcr[:, 1:2, :, :]
            Cr = data_Ycbcr[ :, 2:3, :, :]
            R = Y + 1.402 * (Cr - 0.5)
            G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
            B = Y + 1.772 * (Cb - 0.5)
            data_RGB = torch.cat((R, G, B), dim=1)
            data_RGB = torch.clamp(data_RGB, 0., 1.)  # 确保值范围在 [0, 1]

            sal_input = torch.cat((rgb, thermal, data_RGB), dim=0)
            s_coarse, rgb_map, tma_map, y, s_output = sodnet(sal_input)

            gt_coarse = F.interpolate(gt, (s_coarse.shape[2], s_coarse.shape[3]), mode='bilinear', align_corners=True)
            gt_coarse = torch.cat((gt_coarse, gt_coarse, gt_coarse), dim=0)
            gt_specific = torch.cat((gt, gt), dim=0)

            loss_coarse = criteria_sod(s_coarse, gt_coarse)
            loss_final = criteria_sod(s_output, gt)
            loss_y = criteria_sod(y, gt)
            loss_specific = criteria_sod(torch.cat((rgb_map, tma_map), dim=0), gt_specific)
            loss_weights = sod_hyp["loss_weights"]
            loss_sod = (
                loss_coarse * loss_weights["coarse"]
                + loss_final * loss_weights["final"]
                + loss_specific * loss_weights["specific"]
                + loss_y * loss_weights["auxiliary"]
            )

            loss = loss_sod + fuloss
            loss.backward()
            optimizer.step()
    del train_loader, sodnet, optimizer

    print("epochfitnessDone")
    print()


from skimage.io import imsave


# generate RGB images
def generate_color_images(Fusion_Model, test_loaders, out_paths, iter_save=""):
    print("Generating RGB images...")
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    
    
    eval_copy_Model = copy.deepcopy(Fusion_Model).to(device)
    for m in eval_copy_Model.modules():
        if isinstance(m, DRA1BR):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward
    for m in eval_copy_Model.modules():
        if isinstance(m, Decoder):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward
    eval_copy_Model.eval()
    
    
    for i in range(len(out_paths)):
        if iter_save == "":
            test_out_folder = out_paths[i]
        else:
            test_out_folder=os.path.join(out_paths[i] + "evos", iter_save)
        # Model inference
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
        test_loader  =  get_test_loader(ir_root=test_loaders[i][0], 
                                        vis_root=test_loaders[i][1], 
                                        num_workers=test_loaders[i][2], 
                                        image_size=None,
                                        # image_size=test_loaders[i][3],   ###############################################################################################
                                        )
        with torch.no_grad():
            for (irimage, visimage_rgb, image_name) in tqdm(test_loader, bar_format=TQDM_BAR_FORMAT_2):
            # for (irimage, visimage_rgb, _, _, image_name) in tqdm(test_loader, bar_format=TQDM_BAR_FORMAT_2):
                data_IR = irimage.to(device)
                data_Ycbcr = visimage_rgb.to(device)
                data_VIS = data_Ycbcr[:, 0:1, :, :]

                data_Fuse = eval_copy_Model(data_VIS, data_IR)
                
                Y = data_Fuse * 255.
                Cb = data_Ycbcr[:, 1:2, :, :] * 255.
                Cr = data_Ycbcr[ :, 2:3, :, :] * 255.
                
                R = Y + 1.402 * (Cr - 128)
                G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
                B = Y + 1.772 * (Cb - 128)
                rgb_tensor = torch.cat((B, G, R), dim=1)
                rgb_tensor = torch.clamp(rgb_tensor, 0, 255)  # 确保值范围在 [0, 255]
                
                data_rgb_np = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_path = os.path.join(test_out_folder, f"{os.path.split(image_name[0])[1][:-4]}.png")
                cv2.imwrite(output_path, data_rgb_np)
        del test_loader
    
            
def validationtasks(project_root_list=[], project_name=""):
    validation_hyp = DETECTION_HYP["validation"]
    fits = [0]*len(project_root_list)
    for i in range(len(project_root_list)):
        if ("M3FD" in project_root_list[i]) or ("MSOD" in project_root_list[i]) or ("LLVIP" in project_root_list[i]):
            print(f"evalling {project_root_list[i] + project_name}")
            model = YOLO(project_root_list[i] + project_name + "/weights/best.pt")
            metric = model.val(
                batch=validation_hyp["batch_size"],
                split=validation_hyp["split"],
                plots=validation_hyp["plots"],
                save=validation_hyp["save"],
                workers=validation_hyp["workers"],
                project=validation_hyp["project"],
                mode=validation_hyp["mode"],
                name=validation_hyp["name"],
            )
            # print(f"{project_root_list[i] + project_name} fit:  ", metric.box.map*0.7 + metric.box.map50*0.3)
            print()
            fits[i] = (
                metric.box.map * validation_hyp["map_weight"]
                + metric.box.map50 * validation_hyp["map50_weight"]
            ) * validation_hyp["score_scale"]
            del model
        if ("FMB" in project_root_list[i]) or ("MFNet" in project_root_list[i]) or ("WHU" in project_root_list[i]) or ("potsdam" in project_root_list[i]):
            print(f"evalling {project_root_list[i] + project_name}")
            metrics = test_seg(
                            cfgfile=SEGMENTATION_HYP["base_training"]["config_template"].format(
                                dataset=evodata[i]
                            ),
                            ckpt=glob.glob(project_root_list[i] + project_name + "/best_mIoU*.pth")[0],
                            )
            # print(f"{project_root_list[i] + project_name} fit:  ", metrics['mIoU']*0.7 + metrics['mAcc']*0.3)
            print()
            fits[i] = (
                metrics['mIoU'] * SEGMENTATION_HYP["validation"]["miou_weight"]
                + metrics['mAcc'] * SEGMENTATION_HYP["validation"]["mean_accuracy_weight"]
            )
        if ("VT821" in project_root_list[i]) or ("VT1000" in project_root_list[i]) or ("VT5000" in project_root_list[i]):
            print(f"evalling {project_root_list[i] + project_name}")
            metrics = test_sod(
                            dataset=evodata[i],
                            ckpt=os.path.join(project_root_list[i], project_name, "model_sod.pth"),
                            )
            # print(f"{project_root_list[i] + project_name} fit:  ", metrics['mIoU']*0.7 + metrics['mAcc']*0.3)
            print()
            fits[i] = (metrics)
    print("Fits:", fits)
    print("mean fit: ", sum(fits)/len(fits))
    return sum(fits)/len(fits), fits
            
            
def traintasks(project_root_list=[], from_project_name="", project_name=""):
    detection_train_hyp = DETECTION_HYP["base_training"]
    segmentation_config_template = SEGMENTATION_HYP["base_training"]["config_template"]
    # from_project_name=""
    for i in range(len(project_root_list)):
        if ("M3FD" in project_root_list[i]) or ("MSOD" in project_root_list[i]) or ("LLVIP" in project_root_list[i]):
            print("="*80)
            print("Training: ", evodata[i])
            print("="*80)
            if from_project_name == "":
                model = YOLO(detection_train_hyp["initial_model"], task='detect')
            else:
                model = YOLO(project_root_list[i] + from_project_name + "/weights/best.pt")
            model.train(
                data=detection_train_hyp["data_template"].format(dataset=evodata[i]),
                batch=detection_train_hyp["batch_size"],
                epochs=detection_train_hyp["epochs"],
                imgsz=detection_train_hyp["image_size"],
                close_mosaic=detection_train_hyp["close_mosaic"],
                patience=detection_train_hyp["patience"],
                workers=detection_train_hyp["workers"],
                device=detection_train_hyp["device"],
                project=project_root_list[i],
                name=project_name,
                seed=detection_train_hyp["seed"],
            )
            del model
        if ("FMB" in project_root_list[i]) or ("MFNet" in project_root_list[i]) or ("WHU" in project_root_list[i]) or ("potsdam" in project_root_list[i]):
            print("="*80)
            print("Training: ", evodata[i])
            print("="*80)
            if from_project_name == "":
                train_seg(
                            cfgfile=segmentation_config_template.format(dataset=evodata[i]),
                            work_dir=os.path.join(project_root_list[i] + project_name),
                            load_from=None
                        )
            else:
                train_seg(
                            cfgfile=segmentation_config_template.format(dataset=evodata[i]),
                            work_dir=os.path.join(project_root_list[i] + project_name),
                            load_from=glob.glob(project_root_list[i] + from_project_name + "/best_mIoU*.pth")[0],
                        )
        if ("VT821" in project_root_list[i]) or ("VT1000" in project_root_list[i]) or ("VT5000" in project_root_list[i]):
            print("="*80)
            print("Training: ", evodata[i])
            print("="*80)
            if from_project_name == "":
                train_sod(
                            dataset=evodata[i],
                            root_dir=os.path.join(project_root_list[i], project_name),
                            ckpt=None,
                            )
            else:
                train_sod(
                            dataset=evodata[i],
                            root_dir=os.path.join(project_root_list[i], project_name),
                            ckpt=os.path.join(project_root_list[i], from_project_name, "model_sod.pth"),
                            )
    print("Training Done")
    print()


    
            
def validationModelOnTasks(Fusion_Model, test_loaders, out_paths, project_root_list=[], project_name=""):
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    
    validation_hyp = DETECTION_HYP["validation"]
    fits = [0]*len(project_root_list)
    for i in range(len(project_root_list)):
        print("Generating RGB images...")
        test_out_folder = out_paths[i]
        # Model inference
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
        test_loader = get_test_loader(test_loaders[i][0], test_loaders[i][1], test_loaders[i][2], test_loaders[i][3])
        with torch.no_grad():
            for (irimage, visimage_rgb, image_name) in tqdm(test_loader, bar_format=TQDM_BAR_FORMAT_2):
                data_IR = irimage.to(device)
                data_Ycbcr = visimage_rgb.to(device)
                data_VIS = data_Ycbcr[:, 0:1, :, :]

                data_Fuse = Fusion_Model(data_VIS, data_IR)
                
                Y = data_Fuse * 255.
                Cb = data_Ycbcr[:, 1:2, :, :] * 255.
                Cr = data_Ycbcr[ :, 2:3, :, :] * 255.
                
                R = Y + 1.402 * (Cr - 128)
                G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
                B = Y + 1.772 * (Cb - 128)
                rgb_tensor = torch.cat((B, G, R), dim=1)
                rgb_tensor = torch.clamp(rgb_tensor, 0, 255)  # 确保值范围在 [0, 255]
                
                data_rgb_np = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_path = os.path.join(test_out_folder, f"{os.path.split(image_name[0])[1][:-4]}.png")
                cv2.imwrite(output_path, data_rgb_np)
        del test_loader
        
        
        
        if ("M3FD" in project_root_list[i]) or ("MSOD" in project_root_list[i]) or ("LLVIP" in project_root_list[i]):
            print(f"evalling {project_root_list[i] + project_name}")
            model = YOLO(project_root_list[i] + project_name + "/weights/best.pt")
            metric = model.val(
                batch=validation_hyp["batch_size"],
                split=validation_hyp["split"],
                plots=validation_hyp["plots"],
                save=validation_hyp["save"],
                workers=validation_hyp["workers"],
                project=validation_hyp["project"],
                mode=validation_hyp["mode"],
                name=validation_hyp["name"],
            )
            # print(f"{project_root_list[i] + project_name} fit:  ", metric.box.map*0.7 + metric.box.map50*0.3)
            print()
            fits[i] = (
                metric.box.map * validation_hyp["map_weight"]
                + metric.box.map50 * validation_hyp["map50_weight"]
            ) * validation_hyp["score_scale"]
            del model
        if ("FMB" in project_root_list[i]) or ("MFNet" in project_root_list[i]) or ("WHU" in project_root_list[i]) or ("potsdam" in project_root_list[i]):
            print(f"evalling {project_root_list[i] + project_name}")
            metrics = test_seg(
                            cfgfile=SEGMENTATION_HYP["base_training"]["config_template"].format(
                                dataset=evodata[i]
                            ),
                            ckpt=glob.glob(project_root_list[i] + project_name + "/best_mIoU*.pth")[0],
                            )
            # print(f"{project_root_list[i] + project_name} fit:  ", metrics['mIoU']*0.7 + metrics['mAcc']*0.3)
            print()
            fits[i] = (
                metrics['mIoU'] * SEGMENTATION_HYP["validation"]["miou_weight"]
                + metrics['mAcc'] * SEGMENTATION_HYP["validation"]["mean_accuracy_weight"]
            )
        if ("VT821" in project_root_list[i]) or ("VT1000" in project_root_list[i]) or ("VT5000" in project_root_list[i]):
            print(f"evalling {project_root_list[i] + project_name}")
            metrics = test_sod(
                            dataset=evodata[i],
                            ckpt=os.path.join(project_root_list[i], project_name, "model_sod.pth"),
                            )
            # print(f"{project_root_list[i] + project_name} fit:  ", metrics['mIoU']*0.7 + metrics['mAcc']*0.3)
            print()
            fits[i] = (metrics)
            
            
        if fits[i] < best_fits[i]:
            break
    print("Fits:", fits)
    print("mean fit: ", sum(fits)/len(fits))
    return sum(fits)/len(fits), fits




def main():
    # 0.2712 --> 0.02708
    global Fusion_Model, best_fit, best_fits, iter_name, evodata
    iterations = EXPERIMENT_HYP["iterations"]
    fitness = 0
    for iteration in range(iterations):
        print("="*80)
        print("="*80)
        print("="*80)
        print("Iteration: ", iteration)
        print("="*80)
        print("="*80)
        print("="*80)
        print()
        
        if iteration == 0:
            from_iter_name = ""
        else:
            from_iter_name = "iter" + str(iteration - 1)
            
        iter_name = "iter" + str(iteration)
        
        # if iteration != 0:
        #     print("="*80)
        #     print("="*80)
        #     print("Testing new images...")
        #     print("="*80)
        #     print("="*80)
        #     traintasks(project_root_list, '', iter_name+"_test")
        #     fitness, fits = validationtasks(project_root_list, iter_name+"_test")
            
        print("="*80)
        print("="*80)
        print("Training new images...")
        print("="*80)
        print("="*80)
        traintasks(project_root_list, from_iter_name, iter_name)
            
        fitness, fits = validationtasks(project_root_list, iter_name)
        
        best_fit = fitness
        best_fits = fits
        for project in range(len(project_root_list)):
            print("="*80)
            print("="*80)
            print("Fitting Data: ", evodata[project])
            print("="*80)
            print("="*80)
            print()
            # if i != 0:
            if ("M3FD" in project_root_list[project]) or ("MSOD" in project_root_list[project]) or ("LLVIP" in project_root_list[project]):
                fusionckpt = EvolutionDetectionFusion(iteration, project)     # Evodet
            if (("FMB" in project_root_list[project]) or ("MFNet" in project_root_list[project]) or 
                ("WHU" in project_root_list[project]) or ("potsdam" in project_root_list[project])):
                fusionckpt = EvolutionSegmentationFusion(iteration, project)    # Evoseg
            if ("VT821" in project_root_list[project]) or ("VT1000" in project_root_list[project]) or ("VT5000" in project_root_list[project]):
                fusionckpt = EvolutionSODFusion(iteration, project)    # Evosod
        
        # Creating new training images
        generate_color_images(Fusion_Model, val_loader_list, val_folder_list)
        generate_color_images(Fusion_Model, train_loader_list, train_folder_list)
    
if __name__ == '__main__':
    
    generate_color_images(Fusion_Model, val_loader_list, val_folder_list)
    generate_color_images(Fusion_Model, train_loader_list, train_folder_list)
    main()
