"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from collections import OrderedDict
import torch
from xmodaler.checkpoint import XmodalerCheckpointer
import xmodaler.utils.comm as comm
from xmodaler.config import get_cfg
from xmodaler.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, build_engine
from xmodaler.modeling import add_config

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    #cfg.MODEL.WEIGTHS='./configs/image_caption/cosnet/cosnet_xe.pth'
    trainer = build_engine(cfg)
    trainer.resume_or_load(True)
    args.eval_only=True
    res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.test_evaluator, epoch=-1)




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



def dummy():
    if args.eval_only:
        res = None
        if trainer.val_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.val_data_loader, trainer.val_evaluator, epoch=-1)
        if comm.is_main_process():
            print(res)

        if trainer.test_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.test_evaluator, epoch=-1)
        if comm.is_main_process():
            print(res)
        return res

    return trainer.train()