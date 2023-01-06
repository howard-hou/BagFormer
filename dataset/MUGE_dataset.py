from dataset.utils import pre_caption
import json
import base64
import pickle
import codecs
from io import BytesIO
import os
import random
from pathlib import Path

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_img_from_text(image_base64):
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
    return img


def encode_image_file_to_base64(fn):
    img = Image.open(fn)
    return encode_image_to_base64(img)


def encode_image_to_base64(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_bytes = base64.b64encode(byte_data)  # bytes
    base64_str = base64_bytes.decode()
    return base64_str


def load_item_id2image(path):
    item2image = {}
    for l in open(path):
        item_id, image_base64 = l.strip().split("\t")
        # convert item_id to int
        item_id = int(item_id)
        image = load_img_from_text(image_base64)
        item2image[item_id] = image
    return item2image


def load_t2i_ground_truth(path):
    t2i_ground_truth = []
    for l in open(path):
        j = json.loads(l.strip())
        t2i_ground_truth.append(j)
    return t2i_ground_truth


def base64_to_numpy(base64_str):
    byte_data = codecs.decode(base64_str.encode(), "base64")
    emb_as_np = pickle.loads(byte_data)
    return emb_as_np


class MultimodalRetrievalDataset(Dataset):
    """
    Dataset loader for Multimodal_Retrieval datasets.
    """

    def __init__(self, root, set_name, transform, max_words=30):
        self.root = Path(root)
        self.set_name = set_name
        self.t2i_gt = load_t2i_ground_truth(
            self.root / f"MR_{split}_queries.jsonl")
        self.item_id2image = load_item_id2image(
            self.root / f"MR_{split}_imgs.tsv")
        self.transform = transform
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
