import base64
import json
from io import BytesIO
from PIL import Image


def load_img_from_text(image_base64):
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
    return img.convert("RGB")


def encode_image_file_to_base64(fn):
    img = Image.open(fn)
    return encode_image_to_base64(img)


def encode_image_to_base64(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    return base64_str


def load_item_id2image(path, dtype="int"):
    item_id2img = {}
    for line in open(path, encoding="utf-8"):
        item_id, image_base64 = line.strip().split("\t")
        item_id = ensure_type(dtype, item_id)
        item_id2img[item_id] = image_base64
    return item_id2img


def ensure_type(dtype, item_id):
    # convert id to int or str
    if dtype == "int":
        item_id = int(item_id)
    elif dtype == "str":
        item_id = str(item_id)
    else:
        raise "item id must be int or str"
    return item_id


def load_query_file(path, dtype="int"):
    ground_truth = []
    for l in open(path, encoding="utf-8"):
        j = json.loads(l.strip())
        j["query_id"] = ensure_type(dtype, j["query_id"])
        if "item_ids" in j:
            j["item_ids"] = [ensure_type(dtype, item_id) for item_id in j["item_ids"]]
        ground_truth.append(j)
    return ground_truth

def load_classification_file(path, dtype="int"):
    ground_truth = []
    for l in open(path, encoding="utf-8"):
        j = json.loads(l.strip())
        j["query_id"] = ensure_type(dtype, j["query_id"])
        # from 1-5 to 0-4
        j["relevance"] = ensure_type(dtype, j["relevance"]) - 1
        ground_truth.append(j)
    return ground_truth
