# BagFormer
simple modification for the CLIP code, to get all the visual and textual token from model
```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image_full(image)
    text_features = model.encode_text_full(text)

print("image feature shape:", image_features.shape)  # prints: torch.Size([1, 50, 512])
print("text feature shape:", text_features.shape)  # prints: torch.Size([3, 77, 512])
```