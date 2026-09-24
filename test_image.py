import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from modeling_resnet import ResnetModel

os.makedirs("output", exist_ok=True)

# Load model
model = ResnetModel.from_pretrained("custom-resnet50d")
model.eval()

# 1. Save original image
original = Image.open("testing.png").convert("RGB")
original.save("output/1_original.png")
print("Saved: output/1_original.png")

# 2. Save preprocessed image (resize + crop, before normalization)
preprocess_visual = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
preprocessed = preprocess_visual(original)
preprocessed.save("output/2_preprocessed_224x224.png")
print("Saved: output/2_preprocessed_224x224.png")

# 3. Prepare tensor for model
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transform(original).unsqueeze(0)  # (1, 3, 224, 224)

# Extract features
with torch.no_grad():
    features = model.forward_features(img)               # (1, 2048) pooled
    spatial  = model.forward_features(img, pool=False)   # (1, 2048, 7, 7)

# 4. Save feature map visualization (mean across 2048 channels -> heatmap)
feat_map = spatial[0].mean(dim=0)                        # (7, 7)
feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())  # normalize 0-1
feat_map_img = to_pil_image(feat_map.unsqueeze(0)).resize((224, 224), Image.NEAREST)
feat_map_img.save("output/3_feature_map_heatmap.png")
print("Saved: output/3_feature_map_heatmap.png")

print(f"\nPooled feature vector shape : {features.shape}")
print(f"Spatial feature map shape   : {spatial.shape}")
print(f"First 10 feature values     : {features[0, :10].tolist()}")
print(f"Min: {features.min():.4f}  Max: {features.max():.4f}  Mean: {features.mean():.4f}")
