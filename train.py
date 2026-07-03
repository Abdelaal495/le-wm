import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
import torch.nn.functional as F

from module import SIGReg, StepwiseNormalizedDispersion
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    reg_name = cfg.loss.name
    lambd = cfg.loss[reg_name].weight if reg_name != "none" else 0.0

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]              # usually emb[:, 1:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # Main JEPA predictive loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # Anti-collapse regularizer
    if reg_name == "none":
        output["reg_loss"] = emb.new_tensor(0.0)
    elif reg_name == "sigreg":
        # Original LeWM behavior: SIGReg expects (T, B, D)
        output["reg_loss"] = self.regularizer(emb.transpose(0, 1))
    elif reg_name == "disp":
        # New behavior: dispersion expects (B, T, D)
        output["reg_loss"] = self.regularizer(emb)
    else:
        raise ValueError(f"Unknown regularizer: {reg_name}")

    output["loss"] = output["pred_loss"] + lambd * output["reg_loss"]

    # Extra diagnostics
    with torch.no_grad():
        output["emb_norm"] = emb.float().norm(dim=-1).mean()
        z = F.normalize(emb.float(), dim=-1)
        stepwise_cos = torch.einsum("btd,ctd->tbc", z, z)
        B = emb.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=emb.device).unsqueeze(0)
        output["mean_offdiag_cos"] = stepwise_cos.masked_select(~eye).mean()

    logs = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k in ["emb_norm", "mean_offdiag_cos"]
    }
    self.log_dict(logs, on_step=True, sync_dist=True)

    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    if cfg.loss.name == "sigreg":
        regularizer = SIGReg(**cfg.loss.sigreg.kwargs)
    elif cfg.loss.name == "disp":
        regularizer = StepwiseNormalizedDispersion(**cfg.loss.disp.kwargs)
    elif cfg.loss.name == "none":
        regularizer = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown regularizer: {cfg.loss.name}")
    
    world_model = spt.Module(
        model=world_model,
        regularizer=regularizer,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
