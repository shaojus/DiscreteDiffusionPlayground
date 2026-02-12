import os
import yaml
import subprocess
import wandb
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")

with open("sweep.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

export_vars = f"ALL,SWEEP_ID={sweep_id},WANDB_ENTITY={entity},WANDB_PROJECT={project}"

subprocess.run(["sbatch", f"--export={export_vars}", "run_agent.sh"])

print(f"Sweep {sweep_id} launched for {entity}/{project}")