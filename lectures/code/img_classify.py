import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()  # Set the model to evaluation mode

def classify_image(img, topn=4): 
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
    
    # Make a prediction
    with torch.no_grad():
        output = vgg16(img_tensor)
    
    # Load the class labels (from ImageNet)
    with open("../data/imagenet_classes.txt") as f:
        class_labels = [line.strip() for line in f.readlines()]
    
    # Get the predicted class index
    predicted_idx = torch.argmax(output).item()
    _, indices = torch.sort(output, descending=True)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print("Probabilities calculated")
    
    d = {'Class': [class_labels[idx] for idx in indices[0][:topn]], 
        'Probability score': [np.round(probabilities[0, idx].item(),3) for idx in indices[0][:topn]]}
    print(pd.DataFrame(d))
    # Print the predicted class label
    predicted_label = class_labels[predicted_idx]
    print(f"Predicted class: {predicted_label}")