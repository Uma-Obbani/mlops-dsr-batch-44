
# MLOps-DSR

This project demonstrates how to fine-tune a ResNet18 model (pre-trained on ImageNet) to detect both fresh and rotten versions of fruits using the apples-oranges-bananas dataset from Kaggle. The model's training performance is tracked with Weights & Biases (W&B), and the best-performing model is stored as an artifact on W&B.

**Fine-tuning was performed in this Kaggle Notebook:**  
[https://www.kaggle.com/code/salmanfariznavas/scratchpad-fruit-classifier-5-09-2025](https://www.kaggle.com/code/salmanfariznavas/scratchpad-fruit-classifier-5-09-2025)

The main objective is to provide hands-on experience with a simple MLOps pipeline: training, tracking, and deploying a machine learning model as a FastAPI endpoint on Google Cloud Run.

## What this repo does

- Fine-tunes a ResNet18 model on the apples-oranges-bananas dataset (including detection of rotten fruit).
- Tracks experiment metrics and model artifacts using Weights & Biases.
- Downloads the best model artifact from W&B using a script in `app/model.py`.
- Stores the downloaded artifact in a local `models` directory (configurable in `app/model.py`).
- Serves predictions via a FastAPI app, with endpoints for health check and image classification.
- Demonstrates containerization with Docker and deployment to Google Cloud Run.

## Prerequisites

- Python 3.10+ installed

# MLOps-DSR

Small FastAPI service that loads a trained ResNet model from a W&B artifact and exposes a single prediction endpoint.

## Project structure (important files)

- `app/main.py` — FastAPI app; endpoints: `GET /` (health) and `POST /predict` (image upload -> classification).
- `app/model.py` — model loading, artifact download, and image transforms.
- `requirements.txt` — Python dependencies.

## Prerequisites

- Python 3.10+
- A W&B account and an API key (if you want to download the model artifact)

## Quick start (recommended)

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file at the repository root with these values (replace with your values):

  ```text
 WANDB_API_KEY=your_wandb_api_key
  WANDB_ORG=your_wandb_organization
  WANDB_PROJECT=your_wandb_project
  WANDB_MODEL_NAME=artifact_name
  WANDB_MODEL_VERSION=version_or_alias
  ```

  **Note:** When running in Docker, these environment variables are set in the Dockerfile. You can edit the Dockerfile to add your values, or use `--env` flags with `docker run` to override them at runtime.

## Running with Docker

You can build and run the FastAPI app in a Docker container:

1. **Build the Docker image:**

  ```bash
  docker build -t <name_of_image> .
  ```

2. **Run the Docker container and forward the port:**

  ```bash
  docker run -p 8080:8080 <name_of_image>
  ```

  This will start the FastAPI app inside the container and expose it on port 8080. You can then check the app at [http://localhost:8080/docs](http://localhost:8080/docs).

  **Environment variables:**

- The Dockerfile sets the W&B variables as ENV instructions. You can edit the Dockerfile to set your values, or override them at runtime:

    ```bash
    docker run -p 8080:8080 \
     -e WANDB_API_KEY=your_wandb_api_key \
     -e WANDB_ORG=your_wandb_organization \
     -e WANDB_PROJECT=your_wandb_project \
     -e WANDB_MODEL_NAME=artifact_name \
     -e WANDB_MODEL_VERSION=version_or_alias \
     <name_of_image>
    ```

## Deploying to Google Cloud Run

Once you have tested the Docker image locally, you can deploy it to Google Cloud Run directly from your GitHub repository:

1. **Push your code to GitHub.**
2. **Create a new Cloud Run service** in the Google Cloud Console and connect it to your GitHub repo.
3. **Configure build and deploy settings:**

- Set the build source to your repo and the Dockerfile.
- Set environment variables (WANDB_API_KEY, etc.) in the Cloud Run service settings.


4. **Deploy.**

After deployment, you will get a public URL to access your FastAPI app running on Cloud Run.

## Run the FastAPI app

Use the command you provided to run the app locally (port 8080, auto-reload):

```bash
fastapi run app/main.py --port 8080 --reload
```

Alternative (common):

```bash
uvicorn app.main:app --port 8080 --reload
```

Open <http://localhost:8080/docs> to view the interactive Swagger UI.

## API

- GET /
  - Health check. Returns a simple JSON message.

- POST /predict
  - Accepts a multipart/form-data file field named `input_image`.
  - Response model:

    ```json
    {
      "category": "freshapple",
      "confidence": 0.92
    }
    ```

  - Example curl (replace image path):

    ```bash
    curl -X POST "http://localhost:8080/predict" -F "input_image=@/path/to/image.jpg"
    ```

## Notes about model & transforms

- `app/model.py` currently sets `MODELS_DIR = "../models"`. That path is resolved relative to the process current working directory — so running the server from the repository root will place `models/` one level above the repo. Consider changing `MODELS_DIR` to be relative to the `app/` directory if you want the models folder created inside the repo. Example:

```python
import os
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)
```

- The default transforms in `load_transforms()` resize the shorter side to 256 and then center-crop to 224, so the network input is 224×224. If you want a final 256×256 image, change the pipeline to `Resize((256,256))` or `Resize(256)` + `CenterCrop(256)`.

## Troubleshooting

- If the models folder is not created: print `MODELS_DIR` in `app/model.py` and check permissions.
- If artifact download fails: verify `.env` variables and that `WANDB_API_KEY` is correct.
- If the server doesn't start: ensure `fastapi`/`uvicorn` is installed and you activated the virtualenv.

## Examples & testing

- Try the Swagger UI at `/docs` for quick manual tests.
- Use the curl example above to test the `/predict` endpoint from the command line.

## License

MIT

  WANDB_API_KEY=your_wandb_api_key
  WANDB_ORG=your_wandb_organization
