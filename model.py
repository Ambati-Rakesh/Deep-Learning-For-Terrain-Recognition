# model.py
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=5)
model.load_state_dict(torch.load('terrain_recognition_vit.pth'))
model.to(device)
model.eval()

# Class names 
class_names = ['Grassy_Terrain', 'Marshy_Terrain','Other_Image','Rocky_Terrain', 'Sandy_Terrain']

def predict_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor.to(device)).logits
        _, predicted = torch.max(output, 1)

    return predicted.item()
