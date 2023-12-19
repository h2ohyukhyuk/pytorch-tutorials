# https://tutorials.pytorch.kr/intermediate/dist_tuto.html
# https://csm-kr.tistory.com/47
'''
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=23456 main.py
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=1 --master_addr="127.0.0.1" --master_port=23456 main.py
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=2 --master_addr="127.0.0.1" --master_port=23456 main.py

RANK: global rank, 전체 node에서 process ID
LOCAL_RANK: node 하나 내부에서 process ID
WORLD_SIZE: 전체 node에서 process 개수
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import io
from datetime import datetime
import argparse

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_ex1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )
        self.feature_ex2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=80),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=80),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=80, out_channels=10, kernel_size=3, padding=1)
        )

    def forward(self, x):
        f1 = self.feature_ex1(x)
        f2 = self.feature_ex2(f1)
        logits = torch.mean(f2, dim=(2,3))
        return logits

class MySummaryWriter():
    def __init__(self, path, is_master):
        self.summary = SummaryWriter(path)
        self.is_master = is_master

    def add_text(self, tag, text_string, global_step):
        if self.is_master:
            texts = print_to_string(text_string)
            self.summary.add_text(tag=tag, text_string=texts, global_step=global_step)

    def add_graph(self, model, images):
        if self.is_master:
            self.summary.add_graph(model, images)

    def add_scalars(self, group_name, scalars, global_step):
        if self.is_master:
            self.summary.add_scalars(group_name, scalars, global_step=global_step)

    def add_images(self, name, images, global_step):
        if self.is_master:
            self.summary.add_images(name, images, global_step=global_step)

    def close(self):
        if self.is_master:
            self.summary.close()


def main(args):

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    init_distributed_mode(args)
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    is_master = True if not hasattr(args, 'rank') else True if args.rank == 0 else False

    summary = MySummaryWriter('runs/multi-gpu/%s' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                             is_master)

    device = torch.device(args.device)

    tf_train = transforms.Compose(  [transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                     transforms.Normalize(mean=0.5, std=0.5)])

    tf_test = transforms.Compose(   [transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])

    train_data = MNIST( root='../data', train=True, transform=tf_train, download=True)
    test_data = MNIST(root='../data', train=False, transform=tf_test, download=True)

    if args.distributed:
        train_sampler = DistributedSampler(train_data)
        test_sampler = DistributedSampler(test_data, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
        test_sampler = torch.utils.data.SequentialSampler(test_data)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler) # , num_workers=args.workers
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler) # , num_workers=args.workers

    ce_loss_fn = nn.CrossEntropyLoss()

    model = Net()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    summary.add_text(tag='log', text_string=print_to_string(model), global_step=None)

    opt = torch.optim.AdamW(
        [{'params' : model.feature_ex1.parameters(), 'lr': args.lrs[0]},
         {'params' : model.feature_ex2.parameters()}],
        lr=args.lrs[1], weight_decay=args.weight_decay)

    lr_sch_exp = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    lr_sch_ms = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5, 10], gamma=0.1)

    images, gt = next(iter(test_data_loader))
    summary.add_graph(model_without_ddp, images.to(device))

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        summary.add_scalars('lr', {'group1': lr_sch_ms.get_last_lr()[0], 'group2': lr_sch_ms.get_last_lr()[1]}, global_step=epoch)

        model.train()
        train_losses = []
        for i, (input, target) in enumerate(train_data_loader):
            input, target = input.to(device), target.to(device)
            out = model(input)

            loss = ce_loss_fn(out, target)
            train_losses.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
            if i > 20:
                break

        lr_sch_exp.step()
        lr_sch_ms.step()

        summary.add_images('train_images', input / 2 + 0.5, global_step=epoch)

        model.eval()
        val_losses = []
        top5s = []
        top1s = []
        for i, (input, target) in enumerate(test_data_loader):
            input, target = input.to(device), target.to(device)
            out = model(input)

            loss = ce_loss_fn(out, target)
            val_losses.append(loss.item())

            top_logits, top_idxs = torch.topk(out, k=5, dim=1, largest=True, sorted=True)
            true_positive = (top_idxs == target.unsqueeze(1)).to(torch.float).sum(dim=0) # batch x 5 -> batch
            top5_cnt = true_positive.sum() # batch -> 1
            top1_cnt = true_positive[0] # batch -> 1
            top5s.append(top5_cnt.item())
            top1s.append(top1_cnt.item())

        summary.add_images('val_images', input / 2 + 0.5, global_step=epoch)
        summary.add_scalars('loss', {'train': np.mean(train_losses), 'valid': np.mean(val_losses)}, global_step=epoch)
        num_test_sam = len(test_data_loader.dataset)
        summary.add_scalars('acc', {'top1': sum(top1s)/num_test_sam, 'top5': sum(top5s)/num_test_sam}, global_step=epoch)
        print(epoch, f'acc top1: {sum(top1s)/num_test_sam:0.3f}')

        # save
        '''
        checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
        save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        '''

    summary.close()

def get_args_parser(add_help=True):

    parser = argparse.ArgumentParser(description="PyTorch Training", add_help=add_help)

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=16, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lrs",
        default=[0.01, 0.005],
        nargs='+',
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[5, 10],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=[0.1,0.05], nargs="+", type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )

    parser.add_argument("--output-dir", default="model", type=str, help="path to save outputs")
    parser.add_argument("--log-dir", default="runs/multi-gpu", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    #parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)