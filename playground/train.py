import os, sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
import wandb

from playground.data.gmm_binary_stream import GMMBinaryStream
from playground.utils.metrics import divergence_metrics_plus

def _wandb_setup(cfg):
    if cfg.get('wandb', {}).get('enabled', True):
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True))
    else:
        os.environ['WANDB_DISABLED'] = 'true'

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    L = 2 * int(cfg.data.n_bits)
    ds = GMMBinaryStream(n_mixes=cfg.data.n_mixes, n_bits=cfg.data.n_bits, device=str(device))
    loader = DataLoader(ds, batch_size=cfg.train.batch_size)

    model = instantiate(cfg.model, max_len=L, device=device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=0.1)

    _wandb_setup(cfg)

    it = iter(loader)
    model.train()
    for step in trange(int(cfg.train.steps)):
        x = next(it).to(device).long()
        loss = model.training_loss(x)
        opt.zero_grad(); loss.backward(); opt.step()

        if (step % int(cfg.log.every)) == 0:
            wandb.log({'loss': float(loss)}, step=step)

        if ((step + 1) % int(cfg.eval.interval)) == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                toks = model.sample(int(cfg.eval.n_samples), L, device)
                toks_np = toks.detach().cpu().numpy()
                gen_xy = np.stack([ds.decode(seq) for seq in toks_np], axis=0)
                mets = divergence_metrics_plus(ds, gen_xy)
                wandb.log({f'eval/{k}': v for k, v in mets.items()}, step=step)
            model.train()

    os.makedirs(os.path.dirname(cfg.train.save_path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': OmegaConf.to_container(cfg, resolve=True)}, cfg.train.save_path)
    print(f"Saved checkpoint to {cfg.train.save_path}")

if __name__ == '__main__':
    main()
