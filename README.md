# BagFormer
simple modification for the CLIP code, to get all the visual and textual token from model
## CLIP way: CLS token similarity
```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("corgi.webp")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# CLIP way to calculate similarity
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```
## token-wise similarity
```python
import torch
import clip
from PIL import Image
from bagformer.model_helper import tokenwise_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("corgi.webp")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
with torch.no_grad():
    image_features = model.encode_image_full(image)
    text_features = model.encode_text_full(text)
    logits_per_image = tokenwise_similarity(image_features, text_features)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
print("image feature shape:", image_features.shape)  # prints: torch.Size([1, 50, 512])
print("text feature shape:", text_features.shape)  # prints: torch.Size([3, 77, 512])
print("Label probs:", probs)
```

# BagFormer way to calculate similarity
```python
import clip
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BertTokenizer
from bagformer.model_helper import tokenwise_similarity, EmbeddingBagHelper

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open("corgi.webp")).unsqueeze(0).to(device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedding_bag_helper = EmbeddingBagHelper(tokenizer, "vocab/demo")
max_seq_len = 77
category = ["a diagram", "a dog", "a cat", "a corgi"]
text = tokenizer(category, padding='max_length', max_length=max_seq_len)
embed_bag_offset, attn_mask = embedding_bag_helper.process(text, return_mask=True)
embed_bag_offset = torch.LongTensor(embed_bag_offset).to(device)
embed_bag_attn_mask = torch.LongTensor(attn_mask).to(device)
text = text.convert_to_tensors("pt").to(device)
with torch.no_grad():
    image_features = model.encode_image_full(image)
    text_features = model.encode_text_full(text.input_ids)
    batch_size, seq_len, text_width = text_features.shape
    embedding_input = torch.arange(batch_size * seq_len, device=device)
    embedbag_feats = F.embedding_bag(embedding_input,
                                     text_features.view(-1, text_width),
                                     embed_bag_offset,
                                     mode='sum').view(batch_size, -1, text_width)
    embedbag_feats = F.normalize(embedbag_feats, dim=-1)
    # pad to same length
    embedbag_seq_len = embedbag_feats.shape[1]
    embedbag_feats = F.pad(embedbag_feats, 
                           pad=(0, 0, 0, max_seq_len-embedbag_seq_len, 0, 0),
                           mode='constant', value=0)
    logits_per_image = tokenwise_similarity(image_features, embedbag_feats)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
print("image feature shape:", image_features.shape)  # prints: torch.Size([1, 50, 512])
print("text feature shape:", embedbag_feats.shape)  # prints: torch.Size([3, 77, 512])
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```