import os
import wandb
from loadotenv import load_env 

MODELS_DIR = "../models"

os.makedirs(MODELS_DIR, exist_ok=True)

load_env()
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
api = wandb.Api()

#artifact_path = "username/project_name/artifact_name:version"

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    assert 'WANDB_ORG' in os.environ, "WANDB_ORG not found in environment variables"
    assert 'WANDB_PROJECT' in os.environ, "WANDB_PROJECT not found in environment variables"
    assert 'WANDB_MODEL_NAME' in os.environ, "WANDB_MODEL_NAME not found in environment variables"
    assert 'WANDB_MODEL_VERSION' in os.environ, "WANDB_MODEL_VERSION not found in environment variables"

    # artifact_path ="maheswaree-no/mlops_dsr_batch_44/resnet18:v1"
    artifact_path = f"{os.getenv('WANDB_ORG')}/{os.getenv('WANDB_PROJECT')}/{os.getenv('WANDB_MODEL_NAME')}:{os.getenv('WANDB_MODEL_VERSION')}"
    artifact = api.artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)

# WANDB_ORG=maheswaree-no
# WANDB_PROJECT=mlops_dsr_batch_44
# WANDB_MODEL_NAME=resnet18
# WANDB_MODEL_VERSION=v1