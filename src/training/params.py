import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=0, help="How often to run zero shot."
    )
    parser.add_argument(
        "--report-to",
        default='tensorboard',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv file with training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default='',
        help="Root directory of training images (empty means paths in the csv are used as-is)",
    )
    parser.add_argument(
        "--val-data-root",
        type=str,
        default='',
        help="Root directory of validation images (empty means paths in the csv are used as-is)",
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--warmup", type=int, default=2000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per GPU."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--model",  # 学生模型
        type=str,
        default="ViT-T-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--t-model",  # 教师模型
        type=str,
        default="ViT-B-16",
        help="Name of the teacher vision backbone to use.",
    )
    parser.add_argument(
        "--t-model-checkpoint",  # 教师权重路径
        type=str,
        default=None,
        help="Path to the teacher model checkpoint",
    )
    # FD
    parser.add_argument("--alpha_fd_loss", type=float, default=2000., help="FD_loss weight")
    # PGS
    parser.add_argument("--alpha_pgs_loss", type=float, default=1., help="PGSLoss weight")
    # DAB
    parser.add_argument("--alpha_dab_loss", type=float, default=1., help="DABLoss weight")
    # GEN
    parser.add_argument("--alpha_gen_loss", type=float, default=1., help="GENLoss weight")

    # === 蒸馏层号 & mask 超参 ===
    parser.add_argument(
        "--distill-layers", type=str, default="0,1,6,11",
        help="PGS 蒸馏使用的 Transformer 层号，逗号分隔")
    parser.add_argument(
        "--dab-layers", type=str, default="0,1,6",
        help="DAB 注意力模仿使用的 Transformer 层号，逗号分隔")
    parser.add_argument(
        "--dab-mask-alpha", type=float, default=0.5,
        help="DAB 注意力阈值系数: mean + alpha * std")
    parser.add_argument(
        "--gen-mask-ratio", type=float, default=0.5,
        help="视觉生成损失的 token 遮挡比例")
    parser.add_argument(
        "--tag",  # 实验标签
        type=str,
        default="test",
        help="tag for this try",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )

    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="csv",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--val-dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="csv",
        help="Which type of val dataset to process."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-r",
        type=str,
        default=None,
        help="Path to imagenet-r for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-a",
        type=str,
        default=None,
        help="Path to imagenet-a for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-sketch",
        type=str,
        default=None,
        help="Path to imagenet-sketch for conducting zero shot evaluation.",
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(  # 训练轮数
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")

    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--t-eval",
        default=False,
        action="store_true",
        help="Whether to evaluate the pretrained teacher model"
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        action="store_true",
        help="Enable deterministic cuDNN behavior (disables cudnn.benchmark)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    # 将层号字符串解析为 tuple，例如 "0,1,6,11" → (0, 1, 6, 11)
    args.distill_layers = tuple(int(x.strip()) for x in args.distill_layers.split(','))
    args.dab_layers = tuple(int(x.strip()) for x in args.dab_layers.split(','))

    return args
