import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BertTokenizer
from models.model_retrieval_bagwise import BagFormer
from models.model_helper import EmbeddingBagHelperAutomaton
from models.loss import tokenwise_similarity_martix

device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = "bert-base-chinese"
max_seq_len = 25

tokenizer = BertTokenizer.from_pretrained(
    text_encoder)

model = BagFormer(
    config=config, 
    text_encoder=text_encoder, 
    tokenizer=tokenizer
)
checkpoint = torch.load("path-to-ckpt", map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)

embedding_bag_helper = EmbeddingBagHelperAutomaton(
        tokenizer, 
        config["entity_dict_path"],
        masked_token=["[CLS]", "[PAD]"]
    )

product = ["rumble roller", "nike zoomx vista"]
text = tokenizer(
    product, 
    padding='max_length', 
    max_length=max_seq_len)

embed_bag_offset, attn_mask = embedding_bag_helper.process(
    text, 
    return_mask=True)
embed_bag_offset = torch.LongTensor(embed_bag_offset).to(device)
embed_bag_attn_mask = torch.LongTensor(attn_mask).to(device)
text = text.convert_to_tensors("pt").to(device)

with torch.no_grad():
    image_features = model.visual_encoder(image)
    text_features = model.text_encoder(
            text.input_ids, 
            attention_mask=text.attention_mask, 
            mode="text"
        ).last_hidden_state
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
    # calc bagwise similarity matrix
    sim_i2t, sim_t2i = tokenwise_similarity_martix(
        embedbag_feats,
        image_features)
    
print("image feature shape:", image_features.shape)  # prints: torch.Size([1, 50, 512])
print("text feature shape:", embedbag_feats.shape)  # prints: torch.Size([4, 77, 512])
print("img2text sim:", sim_i2t)  # prints: [0.04407 0.02673 0.04407 0.8853 ]
print("text2img sim:", sim_t2i)  # prints: [0.04407 0.02673 0.04407 0.8853 ]