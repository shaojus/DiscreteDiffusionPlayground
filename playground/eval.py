import os, sys, json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch, numpy as np

from playground.data.gmm_binary_stream import GMMBinaryStream
from playground.utils.metrics import divergence_metrics_plus

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    device = torch.device(cfg.device)
    L = 2 * int(cfg.data.n_bits)

    ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
    mcfg = ckpt.get('cfg', {}).get('model', cfg.model)

    model = instantiate(mcfg, max_len=L, device=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    with torch.no_grad():
        toks = model.sample(int(cfg.eval.n_samples), L, device)

    ds = GMMBinaryStream(n_mixes=cfg.data.n_mixes, n_bits=cfg.data.n_bits, device=str(device))
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
