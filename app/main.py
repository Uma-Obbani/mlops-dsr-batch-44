import torch
import io
from pydantic import BaseModel
from torchvision.models import ResNet
from fastapi import FastAPI, File, UploadFile, Depends
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from torchvision.models import resnet18


categories = ['freshapple', 'freshbanana', 'freshorange', 'rottenapple', 'rottenbanana', 'rottenorange']

# we must define a __init__.py file in the app directory to make it a package


from app.model import load_model ,  load_transforms

class Result(BaseModel):
    category: str
    confidence: float
    
  
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the fruit classification API"}

@app.post('predict/', response_model=Result)
async def predict(
    input_image: UploadFile = File(...)  ,
    model: ResNet = Depends(load_model),
    transforms:transforms.Compose=Depends(load_transforms)
     
    ) -> Result:
    image = Image.open(io.BytesIO(await input_image.read())).convert("RGB")
    image = transforms(image).reshape(1,3,224,224)  # add batch dimension
    model.eval()
    with torch.inference_mode():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        predicted_confidence, predicted_class_idx = torch.max(probs, dim=1)
        predicted_category = categories[predicted_class_idx.item()]
        
        
        
        
    return Result(category=predicted_category, confidence=predicted_confidence.item())
    return Result(category="freshapple", confidence=0.99)

