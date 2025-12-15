import os
import wandb
from loadotenv import load_env 
import torch
import io
from pydantic import BaseModel
from torchvision.models import ResNet
from fastapi import FastAPI, File, UploadFile, Depends
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms

MODELS_DIR = "../models"

os.makedirs(MODELS_DIR, exist_ok=True)

#load_env()
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

def get_raw_model() -> ResNet:
     '''Get the architecture of the model (random weights), this must match the architecture used during training'''
     architecture = resnet18(weights=None)
     architecture.fc = nn.Sequential(
         nn.Linear(in_features=512, out_features=512),
         nn.ReLU(),
         nn.Linear(in_features=512, out_features=6)
     )
     return architecture
 
def load_model() -> ResNet:
    '''Gives us the model with the trained weights'''
    download_artifact()
    # This gets the model architecture with random weights
    model = get_raw_model()
    # This loads the weights from the file into a state dictionary
    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILENAME
    model_state_dict = torch.load(model_state_dict_path, map_location='cpu')
    # This merges the trained weights into the model architecture so that it no longer has random weights
    model.load_state_dict(model_state_dict, strict=True)
    # Turn off Dropout and BatchNorm uses stats from training
    # IMPORTANT: must be done before inference
    model.eval()
    return model

def load_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )










