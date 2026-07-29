import os, sys, json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch, numpy as np

from playground.data import build_binary_2d_dataset
from playground.utils.metrics import divergence_metrics_plus


def _copy_config(config):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    return OmegaConf.create(config)


def _resolve_model_config(config):
    resolved = _copy_config(config)
    if resolved.get('d_hid') is None and resolved.get('d_model') is not None:
        d_hid_factor = int(resolved.get('d_hid_factor', 4))
        resolved.d_hid = int(resolved.d_model) * d_hid_factor
    if 'd_hid_factor' in resolved:
        del resolved['d_hid_factor']
    return resolved


def resolve_checkpoint_config(ckpt, runtime_cfg):
    saved_cfg = ckpt.get('cfg') or {}
    if OmegaConf.is_config(saved_cfg):
        saved_cfg = OmegaConf.to_container(saved_cfg, resolve=True)

    saved_model = saved_cfg.get('model') if hasattr(saved_cfg, 'get') else None
    saved_data = saved_cfg.get('data') if hasattr(saved_cfg, 'get') else None
    saved_seed = saved_cfg.get('seed') if hasattr(saved_cfg, 'get') else None

    model_cfg = _resolve_model_config(
        saved_model if saved_model is not None else runtime_cfg.model
    )
    data_cfg = _copy_config(saved_data if saved_data is not None else runtime_cfg.data)
    if data_cfg.get('encoding') is None:
        data_cfg.encoding = 'binary'

    seed = int(runtime_cfg.seed if saved_seed is None else saved_seed)
    return model_cfg, data_cfg, seed


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    device = torch.device(cfg.device)

    ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
    model_cfg, data_cfg, training_seed = resolve_checkpoint_config(ckpt, cfg)

    torch.manual_seed(training_seed)
    ds = build_binary_2d_dataset(
        data_cfg,
        device=device,
        realized_state=ckpt.get('data_state'),
    )
    L = 2 * int(data_cfg.n_bits)

    model = instantiate(model_cfg, max_len=L, device=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        toks = model.sample(int(cfg.eval.n_samples), L, device)

    toks_np = toks.detach().cpu().numpy()
    gen_xy = np.stack([ds.decode(seq) for seq in toks_np], axis=0)
    mets = divergence_metrics_plus(ds, gen_xy)

    os.makedirs(os.path.dirname(cfg.eval.out_json), exist_ok=True)
    with open(cfg.eval.out_json, 'w') as f:
        json.dump(mets, f, indent=2)
    print('Saved metrics to', cfg.eval.out_json)
    print(json.dumps(mets, indent=2))

if __name__ == '__main__':
    main()
