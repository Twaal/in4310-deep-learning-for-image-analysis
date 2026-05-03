import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config
from nltk_tokenizer.tokenize import word_tokenize
from utils.vocabulary import Vocabulary


class COCODataset(Dataset):
    def __init__(self, annotation_file, frozen_feature_path, config: Config, is_train):
        """
        :param annotation_file: Path toe COCO dataset annotation file.
        :param vocab_file: Path to the saved vocabulary file
        :param frozen_feature_path: Path to pickle file with frozen ResNet50 features.
                                    The pickle file is a dict where the key is the image id
                                    (i.e. file name without '.jpg') and the value is a numpy array.
        :param vocab_size: Maximum size of the vocabulary (use whatever was used to in the saved vocab file).
        :param max_caption_length: Maximum caption length (in tokens) allowed.
        :param is_train: True if the dataset is training dataset.
        """
        self.annotation_file = annotation_file
        self.vocab_file = config.vocabulary_file
        self.frozen_feature_path = frozen_feature_path
        self.max_caption_length = config.max_caption_length

        # Load vocabulary from CSV
        self.vocab = Vocabulary(config.vocabulary_size, config.vocabulary_file)

        # Load the frozen ResNet50 features from pickle
        with open(frozen_feature_path, 'rb') as f:
            self.frozen_features = pickle.load(f)

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.dataset = json.load(f)

        # Build an image index mapping image id -> image info
        # Also create a mapping from image id to file name (without extension) to lookup frozen features.
        self.imgs = {}
        self.img_id_to_name = {}
        for img in self.dataset.get('images', []):
            self.imgs[img['id']] = img
            base_name = os.path.splitext(img['file_name'])[0]
            self.img_id_to_name[img['id']] = base_name

        self.data = []  # Each item will be a dict with keys: feature, caption, mask, image_id
        if is_train:
            # Process annotations: filter captions, tokenize and convert to indices.
            self._process_train_data(config.max_caption_length)
        else:
            self._process_val_data()

    def _process_train_data(self, max_caption_length):
        for ann in self.dataset['annotations']:
            caption = ann['caption'].strip().lower()
            # Ensure caption ends with a period
            if not caption.endswith('.'):
                caption += '.'
            # Tokenize the caption
            tokens = word_tokenize(caption)
            tokens = ['<start>'] + tokens
            # Filter by maximum caption length
            if len(tokens) > max_caption_length:
                continue
            # Filter out captions with words missing from the vocabulary.
            if any(token not in self.vocab.word2idx for token in tokens):
                continue

            # Convert tokens to indices using the vocabulary mapping.
            word_idxs = [self.vocab.word2idx[token] for token in tokens]
            current_num_words = len(word_idxs)
            # Create a fixed-length array for the caption (padded with zeros)
            caption_array = np.zeros(max_caption_length, dtype=np.int32)
            caption_array[:current_num_words] = np.array(word_idxs)

            # Get the image id and look up its file name (without extension)
            img_id = ann['image_id']
            key = self.img_id_to_name.get(img_id, None)
            if key is None or key not in self.frozen_features:
                # Skip if no feature is found for this image.
                continue

            # Load the frozen feature (a numpy array) and convert it to a torch tensor.
            feature = self.frozen_features[key]
            feature_tensor = torch.tensor(feature, dtype=torch.float)

            # Convert caption and mask to tensors.
            caption_tensor = torch.tensor(caption_array, dtype=torch.long)

            # Save the processed sample.
            self.data.append({
                'feature': feature_tensor,
                'caption': caption_tensor,
                'image_id': img_id
            })

    def _process_val_data(self):
        img_to_captions = defaultdict(list)
        for ann in self.dataset['annotations']:
            img_id = ann['image_id']
            caption = ann['caption'].strip()
            if not caption.endswith('.'):
                caption += '.'
            img_to_captions[img_id].append(caption)

        # For each image, create one sample with feature and list of captions.
        for img_id, captions in img_to_captions.items():
            key = self.img_id_to_name.get(img_id, None)
            if key is None or key not in self.frozen_features:
                continue
            feature = self.frozen_features[key]
            feature_tensor = torch.tensor(feature, dtype=torch.float)
            self.data.append({
                'feature': feature_tensor,
                'caption': captions,
                'image_id': img_id
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # Return (image feature, caption indices, caption mask)
        return sample['feature'], sample['caption'], sample['image_id']


def coco_collate_fn(batch):
    """
    Custom collate function for COCODataset.
    For training mode, captions are tensors (fixed-length) and will be stacked.
    For validation mode, captions are raw strings and are collected into a list.
    Image IDs are always returned as a list.
    """
    features, captions, image_ids = zip(*batch)
    features = torch.stack(features, dim=0)

    # Check the type of captions: if they're tensors, we're in training mode.
    if isinstance(captions[0], torch.Tensor):
        captions = torch.stack(captions, dim=0)
    else:
        captions = list(captions)

    return features, captions, list(image_ids)
