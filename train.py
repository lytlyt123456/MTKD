import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import trainers.mtkd

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.caltech101
import datasets.bach
import datasets.brain
import datasets.eyedr


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = ""

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    cfg.DATASET.NUM_SHOTS = args.num_shots

    # new
    if args.second_phase:
        cfg.TRAINER.MTKD.SECOND_PHASE = True

    cfg.TRAINER.MTKD.KD_WEIGHT = args.kd_weight

    if args.reduction:
        cfg.TRAINER.MTKD.REDUCTION = args.reduction

    if args.residual_ratio:
        cfg.TRAINER.MTKD.RESIDUAL_RATIO = args.residual_ratio

    if args.batch_size:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size

    if args.lr:
        cfg.OPTIM.LR = args.lr


def extend_cfg(cfg):
    from yacs.config import CfgNode
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TRAINER.MTKD = CfgNode()
    cfg.TRAINER.MTKD.TEACHER_NAME = 'ViT-L/14'
    cfg.TRAINER.MTKD.PROMPT_LENGTH = 4
    cfg.TRAINER.MTKD.PROMPT_DEPTH = 9
    cfg.TRAINER.MTKD.CTX_INIT = 'a photo of a'
    cfg.TRAINER.MTKD.TEMPERATURE = 1.0
    cfg.TRAINER.MTKD.KD_WEIGHT = 1.0
    cfg.TRAINER.MTKD.REDUCTION = 24
    cfg.TRAINER.MTKD.RESIDUAL_RATIO = 0.2
    cfg.TRAINER.MTKD.SECOND_PHASE = False


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if args.second_phase:
        trainer.load_model("") # model trained in first phase by 5-shot
        trainer.model_add_adapter()

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainers.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument("--num-shots", type=int, default=-1, help="num shots")

    # new
    parser.add_argument("--second-phase", action="store_true", help="second phase")
    parser.add_argument("--kd-weight", type=float, default=1.0, help="KD weight")
    parser.add_argument("--reduction", type=int, default=24, help="reduction")
    parser.add_argument("--residual-ratio", type=float, default=0.2, help="residual ratio")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="LR")

    args = parser.parse_args()
    main(args)
