import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict
from dataset.randaugment import RandomAugment
from .dataset_helper import load_query_file, load_item_id2image, load_img_from_text, load_classification_file


class ProductRelevanceDataset(data.Dataset):
    """
    Dataset loader for product_relevance datasets.
    """

    def __init__(self, query_file, image_file, transform=None):
        self.transform = transform
        self.ground_truth = load_classification_file(query_file)
        self.item_id2image = load_item_id2image(image_file, dtype="str")
        self.prepare_data()

    def prepare_data(self):
        self.img_text_pair = []
        for j in self.ground_truth:
            text = j["query_text"]
            skuid = j["skuid"]
            if skuid not in self.item_id2image:
                continue
            img = self.item_id2image[skuid]
            label = j["relevance"]
            self.img_text_pair.append((skuid, img, text, label))

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        skuid, img, text, label = self.img_text_pair[index]
        img = load_img_from_text(img)
        if self.transform is not None:
            img = self.transform(img)
        return skuid, img, text, label

    def __len__(self):
        return len(self.img_text_pair)


class MultimodalRetrievalDataset(data.Dataset):
    """
    Dataset loader for Multimodal_Retrieval datasets.
    """

    def __init__(self, query_file, image_file, split, dtype, transform=None):
        self.query_file = query_file
        self.image_file = image_file
        self.split = split
        self.dtype = dtype
        self.transform = transform
        if split == "train":
            self.ground_truth = load_query_file(query_file, dtype=self.dtype)
            self.item_id2image = load_item_id2image(image_file, dtype=self.dtype)
            self.prepare_train_data()
        elif split == "valid":
            self.ground_truth = load_query_file(query_file, dtype=self.dtype)
            self.item_id2image = load_item_id2image(image_file, dtype=self.dtype)
            self.prepare_valid_data()
        elif split == "test":
            self.test_query = load_query_file(query_file, dtype=self.dtype)
            self.item_id2image = load_item_id2image(image_file, dtype=self.dtype)
            self.prepare_test_data()
        else:
            raise "wrong split name: only train, valid, test"

    def prepare_train_data(self):
        self.img_text_pair = []
        for j in self.ground_truth:
            text = j["query_text"]
            for item_id in j["item_ids"]:
                img = self.item_id2image[item_id]
                self.img_text_pair.append((img, text))

    def prepare_valid_data(self):
        self.texts = []
        txt_id2txt_idx = {}
        for txt_idx, j in enumerate(self.ground_truth):
            text = j["query_text"]
            txt_id = j["query_id"]
            self.texts.append((text, txt_id))
            txt_id2txt_idx[txt_id] = txt_idx

        self.imgs = []
        img_id2img_idx = {}
        for img_idx, (img_id, img) in enumerate(self.item_id2image.items()):
            img_id2img_idx[img_id] = img_idx
            self.imgs.append((img, img_id))

        self.txt2img = defaultdict(list)
        self.img2txt = defaultdict(list)
        for j in self.ground_truth:
            txt_idx = txt_id2txt_idx[j["query_id"]]
            for img_id in j["item_ids"]:
                img_idx = img_id2img_idx[img_id]
                self.txt2img[txt_idx].append(img_idx)
                self.img2txt[img_idx].append(txt_idx)

    def prepare_test_data(self):
        self.texts = []
        self.txt_idx2txt_id = {}
        for txt_idx, j in enumerate(self.test_query):
            text = j["query_text"]
            txt_id = j["query_id"]
            self.texts.append((text, txt_id))
            self.txt_idx2txt_id[txt_idx] = txt_id

        self.imgs = []
        self.img_idx2img_id = {}
        for img_idx, (img_id, img) in enumerate(self.item_id2image.items()):
            self.img_idx2img_id[img_idx] = img_id
            self.imgs.append((img, img_id))

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        if self.split != "train":
            img, img_id = self.imgs[index]
            img = load_img_from_text(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, img_id

        img, text = self.img_text_pair[index]
        img = load_img_from_text(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, text, index

    def __len__(self):
        if self.split == "train":
            return len(self.img_text_pair)
        return len(self.imgs)


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def get_train_transform(config):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform


def get_test_transform(config):
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform


def get_visualization_transform(image_res):
    visual_transform = transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return visual_transform


def create_dataloader(dataset, config, load_train=True):
    train_transform = get_train_transform(config)

    test_transform = get_test_transform(config)

    if dataset == 'multimodal_retrieval':
        if load_train:
            train_dataset = MultimodalRetrievalDataset(config['train_query_file'],
                                                       config['train_image_file'],
                                                       split='train', dtype="int",
                                                       transform=train_transform)
            train_dataloader = data.DataLoader(train_dataset, batch_size=config["batch_size_train"],
                                               num_workers=config["num_workers_train"], shuffle=True,
                                               pin_memory=False, drop_last=True)
            print("train dataset is loaded")
        else:
            train_dataloader = None
            print("train dataset is not loaded")

        val_dataset = MultimodalRetrievalDataset(config['valid_query_file'],
                                                 config['valid_image_file'],
                                                 split='valid', dtype="int",
                                                 transform=test_transform)
        test_dataset = MultimodalRetrievalDataset(config['test_query_file'],
                                                  config['test_image_file'],
                                                  split='test', dtype="int",
                                                  transform=test_transform)
        val_dataloader = data.DataLoader(val_dataset, batch_size=config["batch_size_test"],
                                         num_workers=config["num_workers_test"], shuffle=False,
                                         pin_memory=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=config["batch_size_test"],
                                          num_workers=config["num_workers_test"], shuffle=False,
                                          pin_memory=False)
        return train_dataloader, val_dataloader, test_dataloader
    elif dataset == "product_relevance":
        train_dataset = ProductRelevanceDataset(config["train_query_file"],
                                                config["train_image_file"],
                                                transform=train_transform)
        train_dataloader = data.DataLoader(train_dataset, batch_size=config["batch_size_train"],
                                           num_workers=config["num_workers_test"], shuffle=True,
                                           pin_memory=False)
        val_dataloader_list = []
        for query_file, image_file in zip(config["test_query_file"], config["test_image_file"]):
            val_dataset = ProductRelevanceDataset(query_file,
                                                  image_file,
                                                  transform=test_transform)
            val_dataloader = data.DataLoader(val_dataset, batch_size=config["batch_size_test"],
                                             num_workers=config["num_workers_test"], shuffle=False,
                                             pin_memory=False)
            val_dataloader_list.append(val_dataloader)
        return train_dataloader, val_dataloader_list, None
    else:
        raise "wrong dataset name"


def create_one_dataloader(query_file, image_file, split, config):
    if split == "train":
        transform = get_train_transform(config)
        batch_size = config["batch_size_train"]
        shuffle = True
        drop_last = True
    else:
        transform = get_test_transform(config)
        batch_size = config["batch_size_test"]
        shuffle = False
        drop_last = False

    dataset = MultimodalRetrievalDataset(query_file,
                                         image_file,
                                         split=split, dtype="str",
                                         transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                 num_workers=config["num_workers_train"],
                                 shuffle=shuffle,
                                 pin_memory=False,
                                 drop_last=drop_last)
    return dataloader
