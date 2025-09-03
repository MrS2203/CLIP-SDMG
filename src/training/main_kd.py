import logging
import os
import sys
import random
import json
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import wandb
except ImportError:
    wandb = None

from src.training.params import parse_args
from src.training.logger import setup_logging
from src.open_clip import create_kd_model_and_transforms, get_tokenizer
from src.open_clip import KDClipLoss
from src.training.data import get_data
from src.training.scheduler import cosine_lr
from src.training.train import train_kd_one_epoch, evaluate


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    args = parse_args(args)  # 训练的相关信息

    if torch.cuda.is_available():  # 判断GPU可用
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # 设置GPU
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # 模型参数信息路径
    configs_path = 'D:\Project\CLIP-KD\src\open_clip\model_configs'

    # 获取教师模型参数信息
    config_path = os.path.join(configs_path, f'{args.t_model}.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        args.t_embed_dim = config['embed_dim']
        args.t_patch_vis = config['vision_cfg']['width']
        args.t_token_txt = config['text_cfg']['width']

    # 获取学生模型参数信息
    config_path = os.path.join(configs_path, f'{args.model}.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        args.s_embed_dim = config['embed_dim']
        args.patch_vis = config['vision_cfg']['width']
        args.token_txt = config['text_cfg']['width']

    # 设置本次实验的名称
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"t_model_{args.t_model}",
            f"s_model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"tag_{args.tag}"  # 实验标签
        ])

    # 创建日志文件
    args.log_path = None
    log_base_path = os.path.join(args.logs, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    args.log_path = os.path.join(log_base_path, 'out.log')
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists."
        )
        return -1

    # 设置日志级别
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    # 单GPU训练
    logging.info(f'Running with a single process. Device {args.device}.')

    # 设置随机数种子
    random_seed(args.seed)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # 加载学生模型、教师模型、训练预处理、验证预处理
    model, t_model, preprocess_train, preprocess_val = create_kd_model_and_transforms(
        args.model,
        args.t_model,
        args.pretrained,
        precision=args.precision,
        device=args.device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,  # 训练数据集的均值
        image_std=args.image_std,  # 训练数据集的方差
    )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    # 模型的参数信息
    logging.info("Teacher Visual Params:")
    logging.info(f"{str(sum([i.numel() for i in t_model.visual.parameters()]) / 1e6)}M")
    logging.info("Teacher Text Params:")
    logging.info(f"{str(sum([i.numel() for i in t_model.transformer.parameters()]) / 1e6)}M")
    logging.info("Student Visual Params:")
    logging.info(f"{str(sum([i.numel() for i in model.visual.parameters()]) / 1e6)}M")
    logging.info("Student Text Params:")
    logging.info(f"{str(sum([i.numel() for i in model.transformer.parameters()]) / 1e6)}M")
    params_file = os.path.join(args.logs, args.name, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    # 加载教师模型
    t_model.eval()
    for t_n, t_p in t_model.named_parameters():
        t_p.requires_grad = False
    checkpoint = torch.load(args.t_model_checkpoint, map_location='cpu')
    """
    ViT_B_16_cc3m_12m_ep32.pt使用sd = checkpoint["state_dict"]
    ViT_B_16-laion400m_e32.pt使用sd = checkpoint
    """
    sd = checkpoint
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    t_model.load_state_dict(sd)
    print('Teacher model loaded successfully')

    # 损失
    loss = KDClipLoss(
        args=args,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        use_horovod=args.horovod).cuda()

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        # 定义排除（no weight decay）条件：1-D 参数（如标量），或名称中包含 bn（BatchNorm）、ln（LayerNorm）、bias、logit_scale
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        # 定义包含（需要 weight decay）的条件：即不满足 exclude 的那些参数
        include = lambda n, p: not exclude(n, p)

        # 把模型所有可训练参数和它们的名字收集成列表
        named_parameters = list(model.named_parameters())
        # 按条件筛选出“gain_or_bias_params”——不做 weight decay 的参数
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        # 剩下的、需要做 weight decay 的参数
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        # 构造 AdamW 优化器，分成三组参数：
        #  1. gain_or_bias_params：weight_decay=0
        #  2. rest_params：weight_decay=args.wd
        #  3. loss.parameters()：通常不希望对 loss module 的参数做 weight decay
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
                {"params": loss.parameters()}
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        # 如果使用 AMP（混合精度），则启用 GradScaler 来自动动态调整梯度缩放，否则用 None
        scaler = GradScaler() if args.precision == "amp" else None

    # 设置训练轮次  （如果需要断点续训，则加载最新轮次的权重）
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # 初始化数据集
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'

    # 训练的情况下 创建学习率调度函数
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none'
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    # 教师测试跨模态图文检索性能
    if args.t_eval:
        print('evaluate teacher:')
        evaluate(t_model, data, start_epoch, args, writer)

    for epoch in range(start_epoch, args.epochs):
        logging.info(f'Start epoch {epoch}')

        train_kd_one_epoch(model, t_model, data, epoch, loss, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints. 保存节点
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    logging.info(f'The files are saved at {args.logs}')
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
