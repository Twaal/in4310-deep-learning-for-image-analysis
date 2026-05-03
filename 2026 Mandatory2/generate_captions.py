import json
import os
from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Config
from eval_metrics.cider.cider import Cider
from model import ImageCaptionModel
from utils.coco_feature_extractor import ResNet18Encoder
from utils.vocabulary import Vocabulary


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform, original_img_transform, annotation_file):
        self.root_dir = root_dir
        self.transform = transform
        self.original_img_transform = original_img_transform
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        with open(annotation_file, 'r') as f:
            self.dataset = json.load(f)

        self.imgs = {}
        for img in self.dataset.get('images', []):
            self.imgs[img['id']] = img
        self.data = []  # Each item is a dict with keys: 'feature' (or 'img_path'), 'caption', 'image_id'
        self._process_val_data()

    def _process_val_data(self):
        img_to_captions = defaultdict(list)
        for ann in self.dataset['annotations']:
            img_id = ann['image_id']
            caption = ann['caption'].strip()
            if not caption.endswith('.'):
                caption += '.'
            img_to_captions[img_id].append(caption)

        for img_id, captions in img_to_captions.items():
            if img_id not in self.imgs:
                continue
            file_name = self.imgs[img_id]['file_name']
            file_path = os.path.join(self.root_dir, file_name)

            self.data.append({
                'file_path': file_path,
                'caption': captions,
                'image_id': img_id
            })

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Load the image from disk.
        image = Image.open(sample['file_path']).convert('RGB')
        orig_image = image.copy()

        image = self.transform(image)
        orig_image = self.original_img_transform(orig_image)
        return image, orig_image, sample['caption'], sample['image_id']


def image_folder_collate_fn(batch):
    images, original_images, captions, image_ids = zip(*batch)
    images = torch.stack(images, dim=0)

    original_images = list(original_images)
    captions = list(captions)
    image_ids = list(image_ids)

    return images, original_images, captions, image_ids


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    try:
        print(f'Loading model from checkpoint {ckpt_path}')
        model.load_state_dict(checkpoint['model'], strict=False)
    except RuntimeError as e:
        print(e)


def generate_captioned_images(ckpt_path, src_dir, dest_dir, max_images=500):
    """
    Processes each image in the source directory, generates a caption using the image captioning model,
    overlays the caption on the image, and saves the result in the destination directory.
    :param ckpt_path: Path to the saved model file
    :param src_dir: Path to the directory containing source images.
    :param dest_dir: Path to the directory to save captioned images.
    :param max_images: (Roughly) maximum number of images to process from teh src directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    config = Config()
    vocab = Vocabulary(config.vocabulary_size, config.vocabulary_file)
    start_token = vocab.word2idx['<start>']

    # Set up the feature extractor
    feature_extractor = ResNet18Encoder()
    feature_extractor.to(device)
    feature_extractor.eval()

    model = ImageCaptionModel(feature_extractor.dim, config.embedding_size, config.hidden_size,
                              config.vocabulary_size, config.max_caption_length,
                              num_layers=2, cell_type='LSTM', use_attention=True)
    load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()

    # Define image transformations (same as used in the training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    original_img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    # Create the dataset and dataloader
    dataset = ImageFolderDataset(src_dir, transform, original_img_transform, config.val_caption_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=10, pin_memory=True,
                            collate_fn=image_folder_collate_fn)

    with torch.no_grad():
        image_count = 0
        for image_tensors, original_images, captions, image_ids in dataloader:
            batch_size = image_tensors.size(0)
            image_tensors = image_tensors.to(device)

            # Extract CNN features.
            cnn_features = feature_extractor(image_tensors)

            # Create a dummy input: a column of start tokens.
            input_tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

            # Run the captioning model in inference mode.
            logits, alphas = model(cnn_features, input_tokens, is_train=False)
            # Decode by taking argmax over vocabulary dimension.
            predicted_idxs = torch.argmax(logits, dim=-1)  # shape: [batch, seq_len]

            # Process each image in the batch.
            for i in range(batch_size):
                # Convert predicted indices to a caption string.
                pred_idxs = predicted_idxs[i].tolist()
                generated_caption = vocab.get_sentence(pred_idxs)

                # Create a figure with the original image and caption title.
                fig, ax = plt.subplots()
                ax.imshow(original_images[i])
                ax.axis('off')
                ax.set_title(generated_caption, fontsize=12)

                # Save the figure with the same filename to the destination directory.
                save_path = os.path.join(dest_dir, f'{image_ids[i]}.jpg')
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

            image_count += batch_size
            if image_count >= max_images:
                print(f'Number of images processed: {image_count} while max images requested: {max_images}. Exiting.')
                break


def resize_attention_map(alpha_map, final_size, smooth=True):
    """
    Resizes a 2D attention map using PIL.

    Args:
        alpha_map (np.array): 2D numpy array (e.g. shape [7,7]).
        final_size (int): Desired output size (final_size x final_size).
        smooth (bool): If True, use bilinear interpolation; otherwise, use nearest neighbor.

    Returns:
        np.array: Resized attention map as a float array in [0,1].
    """
    # Scale the attention map to 0-255 and convert to uint8 for PIL.
    alpha_uint8 = (alpha_map * 255).astype(np.uint8)
    pil_img = Image.fromarray(alpha_uint8)
    # Choose interpolation method.
    resample = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    pil_resized = pil_img.resize((final_size, final_size), resample=resample)
    # Convert back to float [0, 1]
    return np.array(pil_resized).astype(np.float32) / 255.0


def visualize_caption_heatmap(image, caption, alphas,
                              feature_grid_size=7,  # e.g. 7x7 grid
                              final_size=224,
                              smooth=True,
                              cmap='Greys_r',
                              save_path='captioned_image.png'):
    """
    Visualizes the caption by overlaying the attention heatmap for each word and saves the result.

    Args:
        image (PIL.Image): The original image.
        caption (list[str]): List of words in the generated caption.
        alphas (list[Tensor]): List of attention weights for each time step.
                                Each tensor should be of shape [num_regions] (e.g., 49 for 7x7).
        feature_grid_size (int): Spatial size of the feature grid (default: 7 for 7x7).
        final_size (int): The size (width and height) to resize the image to (default: 224).
        smooth (bool): Whether to smooth the attention map using bilinear interpolation.
        cmap (str): Colormap to use for the attention overlay.
        save_path (str): Where to save the resulting figure.
    """
    # Determine number of words to visualize.
    num_words = min(len(caption), len(alphas))
    num_cols = 5
    num_rows = ceil((num_words + 1) / num_cols)

    # Create a matplotlib figure.
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    ax = plt.subplot(num_rows, num_cols, 1)
    plt.title('<start>', fontsize=12)
    plt.imshow(image)
    plt.axis('off')

    for t in range(num_words):
        # Get the attention weights for time step t and reshape to grid.
        # Assuming each alpha is a torch.Tensor of shape [num_regions]
        current_alpha = alphas[t].detach().cpu().numpy().reshape(feature_grid_size, feature_grid_size)
        # Resize the attention map to the final image size.
        alpha_resized = resize_attention_map(current_alpha, final_size, smooth=smooth)

        ax = plt.subplot(num_rows, num_cols, t + 2)
        plt.title(caption[t], fontsize=12)
        plt.imshow(image)
        plt.imshow(alpha_resized, alpha=0.6, cmap=cmap)
        plt.axis('off')

    plt.tight_layout()
    # Save the figure to disk.
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def generate_captioned_images_top_bottom(ckpt_path, src_directory, dest_dir, k=10):
    """
    Processes the entire validation set, generates captions using the model,
    computes the CIDEr score per image, and visualizes the top k images with the highest
    CIDEr scores and bottom k images with the lowest CIDEr scores.

    :param ckpt_path: Path to the saved model checkpoint.
    :param dest_dir: Directory where the visualizations will be saved.
    :param k: Number of images to select from each end of the score spectrum.
    """
    # Setup device and configuration.
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    config = Config()

    # Define image transformation (should match what was used during training).
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    original_img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    # Create the validation dataset and dataloader.
    val_dataset = ImageFolderDataset(src_directory, transform, original_img_transform, config.val_caption_file)
    val_loader = DataLoader(val_dataset, batch_size=256, collate_fn=image_folder_collate_fn,
                            shuffle=False, num_workers=10, pin_memory=True)

    # Set up the feature extractor and the model
    feature_extractor = ResNet18Encoder()
    feature_extractor.to(device)
    feature_extractor.eval()
    model = ImageCaptionModel(feature_extractor.dim, config.embedding_size, config.hidden_size,
                              config.vocabulary_size, config.max_caption_length, 2, 'LSTM', config.use_attention)
    model.to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()

    # Prepare to collect generated captions and ground-truth references.
    all_gts = {}  # image_id -> list of reference captions
    all_res = {}  # image_id -> [generated caption]
    results = []  # list to store per-image info for later visualization
    vocab = Vocabulary(config.vocabulary_size, config.vocabulary_file)
    start_token = vocab.word2idx['<start>']

    # Iterate over the validation loader.
    for images, original_images, captions, image_ids in val_loader:
        images = images.to(device)
        batch_size = images.size(0)
        # Create a dummy input: a column of start tokens.
        input_tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Run the model in inference mode.
        cnn_features = feature_extractor(images)
        logits, alphas = model(cnn_features, input_tokens, is_train=False)
        predicted_idxs = torch.argmax(logits, dim=-1)  # shape: [batch, seq_len]

        # Process each image in the batch.
        for i, img_id in enumerate(image_ids):
            pred_idxs = predicted_idxs[i].tolist()
            generated_caption = vocab.get_sentence(pred_idxs)
            all_res[img_id] = [generated_caption]
            all_gts[img_id] = captions[i]

            # Convert the transformed image back to PIL image for visualization.
            orig_image = original_images[i]
            caption_words = [vocab.words[i] for i in pred_idxs]
            if caption_words[-1] != '.':
                caption_words.append('.')
            length = np.argmax(np.array(caption_words) == '.') + 1
            caption_words = caption_words[:length]
            alphas_for_image = [alpha[i] for alpha in alphas]

            results.append({
                'img_id': img_id,
                'filename': f'{img_id}.jpg',  # Assuming image_ids are filenames
                'orig_image': orig_image,
                'caption': generated_caption,
                'caption_words': caption_words,
                'alphas': alphas_for_image,
            })

    # Compute CIDEr scores using only the CIDEr metric.
    cider_metric = Cider()
    overall_score, per_image_scores = cider_metric.compute_score(all_gts, all_res)
    print('Overall CIDEr score is', overall_score)

    # Map image_id to its computed CIDEr score.
    score_dict = {}
    for idx, key in enumerate(all_res.keys()):
        score_dict[key] = per_image_scores[idx]

    # Add the CIDEr score to each result.
    for r in results:
        r['cider'] = score_dict.get(r['img_id'], 0)

    # Sort results by CIDEr score.
    results_sorted = sorted(results, key=lambda x: x['cider'])
    bottom_k = results_sorted[:k]
    top_k = results_sorted[-k:]

    # Create directories for top and bottom images.
    top_dest = os.path.join(dest_dir, 'top')
    bottom_dest = os.path.join(dest_dir, 'bottom')
    os.makedirs(top_dest, exist_ok=True)
    os.makedirs(bottom_dest, exist_ok=True)

    # Visualize and save the top k (highest CIDEr) images.
    for r in top_k:
        save_path = os.path.join(top_dest, r['filename'])
        visualize_caption_heatmap(r['orig_image'], r['caption_words'], r['alphas'],
                                  feature_grid_size=7, final_size=224, smooth=True,
                                  save_path=save_path)

    # Visualize and save the bottom k (lowest CIDEr) images.
    for r in bottom_k:
        save_path = os.path.join(bottom_dest, r['filename'])
        visualize_caption_heatmap(r['orig_image'], r['caption_words'], r['alphas'],
                                  feature_grid_size=7, final_size=224, smooth=True,
                                  save_path=save_path)

    print(f'Visualizations saved in {dest_dir}')


if __name__ == "__main__":
    _config = Config()
    src_directory = _config.val_images_dir
    ckpt_file_path = 'ckpts/your_file_name'
    dest_directory = f'./captioned_images/{Path(ckpt_file_path).stem}'
    # generate_captioned_images(ckpt_file_path, src_directory, dest_directory, 128)
    generate_captioned_images_top_bottom(ckpt_file_path, src_directory, dest_directory, 100)
