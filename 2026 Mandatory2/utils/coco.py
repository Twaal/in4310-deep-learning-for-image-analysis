__author__ = 'tylin'
__version__ = '2.0'

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json

from nltk_tokenizer.tokenize import word_tokenize


class COCO:
    def __init__(self, annotation_file):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = {}
        self.cats = {}
        self.img_name_to_id = {}

        dataset = json.load(open(annotation_file, 'r'))
        self.dataset = dataset
        self.process_dataset()
        self.create_index()

    def create_index(self):
        print('creating index...')
        anns = {}
        imgToAnns = {}
        cat_to_imgs = {}
        cats = {}
        imgs = {}
        img_name_to_id = {}

        if 'annotations' in self.dataset:
            imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
            anns = {ann['id']: [] for ann in self.dataset['annotations']}
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            imgs = {im['id']: {} for im in self.dataset['images']}
            for img in self.dataset['images']:
                imgs[img['id']] = img
                img_name_to_id[img['file_name']] = img['id']

        if 'categories' in self.dataset:
            cats = {cat['id']: [] for cat in self.dataset['categories']}
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
            cat_to_imgs = {cat['id']: [] for cat in self.dataset['categories']}
            for ann in self.dataset['annotations']:
                cat_to_imgs[ann['category_id']] += [ann['image_id']]

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats
        self.img_name_to_id = img_name_to_id

    def process_dataset(self):
        for ann in self.dataset['annotations']:
            q = ann['caption'].lower()
            if q[-1] != '.':
                q = q + '.'
            ann['caption'] = q

    def filter_by_cap_len(self, max_cap_len):
        print("Filtering the captions by length...")
        keep_ann = {}
        keep_img = {}
        for ann in self.dataset['annotations']:
            if len(word_tokenize(ann['caption'])) <= max_cap_len:
                keep_ann[ann['id']] = keep_ann.get(ann['id'], 0) + 1
                keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1

        self.dataset['annotations'] = [ann for ann in self.dataset['annotations'] if keep_ann.get(ann['id'], 0) > 0]
        self.dataset['images'] = [img for img in self.dataset['images'] if keep_img.get(img['id'], 0) > 0]

        self.create_index()

    def filter_by_words(self, vocab):
        print("Filtering the captions by words...")
        keep_ann = {}
        keep_img = {}
        for ann in self.dataset['annotations']:
            keep_ann[ann['id']] = 1
            words_in_ann = word_tokenize(ann['caption'])
            for word in words_in_ann:
                if word not in vocab:
                    keep_ann[ann['id']] = 0
                    break
            keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1

        self.dataset['annotations'] = [ann for ann in self.dataset['annotations'] if keep_ann.get(ann['id'], 0) > 0]
        self.dataset['images'] = [img for img in self.dataset['images'] if keep_img.get(img['id'], 0) > 0]

        self.create_index()

    def all_captions(self):
        return [ann['caption'] for ann_id, ann in self.anns.items()]
