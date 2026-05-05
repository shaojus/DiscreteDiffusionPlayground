import os, json
import hydra
from omegaconf import DictConfig
import torch, numpy as np

from playground.data.checkerboard import CheckerboardBinaryStream
from playground.utils.metrics import divergence_metrics_plus, get_true_samples

import playground.data.checkerboard as checkerboard_module
import playground.utils.metrics as metrics_module

@hydra.main(version_base=None, config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    device = torch.device(cfg.device)

    if hasattr(cfg, "seed"):
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

    ds = CheckerboardBinaryStream(
        n_cells=cfg.data.n_cells,
        n_bits=cfg.data.n_bits,
        device=str(device),
    )
    print("checkerboard module:", checkerboard_module.__file__)
    print("metrics module:", metrics_module.__file__)
    print("ds type:", type(ds))
    print("has ds.dist:", hasattr(ds, "dist"))
    print("ds attrs:", ds.__dict__.keys())

    gen_xy = get_true_samples(
        ds,
        n=int(cfg.eval.n_samples),
    )
    gen_xy = gen_xy.astype(np.float64)

    mets = divergence_metrics_plus(ds, gen_xy)

    out_json = cfg.eval.out_json
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_json, 'w') as f:
        json.dump(mets, f, indent=2)

    print('Saved true-distribution baseline metrics to', out_json)
    print(json.dumps(mets, indent=2))


if __name__ == '__main__':
    main()