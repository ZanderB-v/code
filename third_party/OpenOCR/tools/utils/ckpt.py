import os

import torch

from tools.utils.logging import get_logger

def save_ckpt(
    model,
    cfg,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    metrics,
    is_best=False,
    logger=None,
    prefix=None,
):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
    """
    if logger is None:
        logger = get_logger()
    if prefix is None:
        if is_best:
            save_path = os.path.join(cfg["Global"]["output_dir"], "best.pth")
        else:
            save_path = os.path.join(cfg["Global"]["output_dir"], "latest.pth")
    else:
        save_path = os.path.join(cfg["Global"]["output_dir"], prefix + ".pth")
    state_dict = model.module.state_dict() if cfg["Global"]["distributed"] else model.state_dict()
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": state_dict,
        "optimizer": None if is_best else optimizer.state_dict(),
        "scheduler": None if is_best else lr_scheduler.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }
    torch.save(state, save_path)
    logger.info(f"save ckpt to {save_path}")


def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None):
    """
    Resume from saved checkpoints
    :param checkpoint_path: Checkpoint path to be resumed
    """
    if logger is None:
        logger = get_logger()
    checkpoints = cfg["Global"].get("checkpoints")
    pretrained_model = cfg["Global"].get("pretrained_model")

    status = {}
    if checkpoints and os.path.exists(checkpoints):
        checkpoint = torch.load(checkpoints, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"resume from checkpoint {checkpoints} (epoch {checkpoint['epoch']})")

        status["global_step"] = checkpoint["global_step"]
        status["epoch"] = checkpoint["epoch"] + 1
        status["metrics"] = checkpoint["metrics"]
    elif pretrained_model and os.path.exists(pretrained_model):
        load_pretrained_params(model, pretrained_model, logger)
        logger.info(f"finetune from checkpoint {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def load_pretrained_params(model, pretrained_model, logger):
    if pretrained_model.endswith(".safetensors"):
        from safetensors.torch import load_file
        logger.info(f"Loading weights from safetensors: {pretrained_model}")
        checkpoint = load_file(pretrained_model)
    else:
        logger.info(f"Loading weights using torch.load: {pretrained_model}")
        checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"))

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    compatible_state = {}
    unexpected_keys = []
    shape_mismatches = []
    for name, value in state_dict.items():
        if name not in model_state:
            unexpected_keys.append(name)
            continue
        if tuple(value.shape) != tuple(model_state[name].shape):
            shape_mismatches.append(
                {
                    "name": name,
                    "pretrained_shape": tuple(value.shape),
                    "model_shape": tuple(model_state[name].shape),
                }
            )
            continue
        compatible_state[name] = value

    missing_keys = [
        name for name in model_state if name not in compatible_state
    ]
    model.load_state_dict(compatible_state, strict=False)
    logger.info(
        "Pretrained compatibility summary: "
        f"loaded={len(compatible_state)}, "
        f"shape_mismatches={len(shape_mismatches)}, "
        f"missing={len(missing_keys)}, "
        f"unexpected={len(unexpected_keys)}"
    )
    for item in shape_mismatches:
        logger.info(
            "Skip pretrained tensor with incompatible shape: "
            f"{item['name']} pretrained={item['pretrained_shape']} "
            f"model={item['model_shape']}"
        )
    for name in missing_keys:
        if name not in state_dict:
            logger.info(f"{name} is not in pretrained model")
    for name in unexpected_keys:
        logger.info(f"{name} from pretrained model is not used")
