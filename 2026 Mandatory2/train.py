import os
import random
import time
from collections import defaultdict
from datetime import timedelta, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from eval_metrics.bleu.bleu import Bleu
from eval_metrics.cider.cider import Cider
from eval_metrics.rogue.rouge import Rouge
from evaluate import evaluate_model
from model import ImageCaptionModel
from utils.dataset import COCODataset, coco_collate_fn
from config import Config
from utils.plot import plot_metrics, plot_loss


def train():
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    print("Training on device ", device)
    config = Config()
    print(vars(config))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=(0.3, 1.3), contrast=(0.3, 1.3), saturation=(0.3, 1.3), hue=(-0.2, 0.2)),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    print(transform)

    # Create datasets
    train_dataset = COCODataset(config.train_caption_file, config.resnet50_features_train_file, config, True)
    val_dataset = COCODataset(config.val_caption_file, config.resnet50_features_val_file, config, False)
    # Create dataloaders from the datasets
    num_workers = 0 if os.name == 'nt' else 10  # Windows doesn't support multiprocessing in DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=coco_collate_fn,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, collate_fn=coco_collate_fn,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    # Instantiate the model, optimiser, and loss function.
    model = ImageCaptionModel(512, config.embedding_size, config.hidden_size, config.vocabulary_size,
                              config.max_caption_length, config.num_layers, config.cell_type, config.use_attention)
    model.to(device)
    # Optimiser and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    # Loss function. Nore the ignore_index.
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.word2idx['<pad>']).to(device)

    val_metrics = [Bleu(4), Cider(), Rouge()]
    global_step = 0
    epoch_start_steps = []
    step_losses = []
    metric_scores_epochs = defaultdict(list)
    best_cider = 0
    best_bleu4 = 0

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _user = os.environ.get('USER') or os.environ.get('USERNAME', 'user')
    model_name = f"{_user}_{current_time}.pth"
    ckpt_path = Path(f'./ckpts/{model_name}')

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        epoch_start_steps.append(global_step)
        epoch_start = time.time()
        print("Starting epoch", epoch, flush=True)
        for features, captions, image_ids in train_loader:
            # Print progress of current epoch
            print(f"Epoch {epoch}, Step {global_step}/{len(train_loader)}, {100*(global_step/len(train_loader)):.2f}%", flush=True)
            # Teacher forcing: shift the captions.
            # For example, if captions are [<start>, token1, token2, ..., ., <PAD>, <PAD>],
            # then inputs are all tokens except the last, and targets are all tokens except the first.
            features = features.to(device)
            input_tokens = captions[:, :-1].to(device)   # (batch_size, max_caption_length - 1)
            target_tokens = captions[:, 1:].to(device)  # (batch_size, max_caption_length - 1)

            # Forward pass.
            outputs, alphas = model(features, input_tokens, True)  # (batch_size, seq_len, vocab_size)

            # Reshape outputs and targets to compute the loss.
            outputs = outputs.reshape(-1, outputs.size(2))  # (batch_size*(seq_len), vocab_size)
            target_tokens = target_tokens.reshape(-1)  # (batch_size*(seq_len))

            loss = criterion(outputs, target_tokens)
            running_loss += loss.item() * input_tokens.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.)
            optimizer.step()

            global_step += 1
            step_losses.append(loss.item())
        print(f'Epoch {epoch} took {timedelta(seconds=time.time() - epoch_start)}', flush=True)
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {running_loss / (len(train_dataset)): .4f}", flush=True)

        # Evaluate the model on the validation data
        eval_start = time.time()
        metric_scores, generated_captions = evaluate_model(val_loader, model, device, val_metrics)
        print(f'Eval after epoch {epoch} took {timedelta(seconds=time.time() - eval_start)}', flush=True)
        for name, (global_score, sentence_scores) in metric_scores.items():
            print(f"{name}: {global_score}", flush=True)
            metric_scores_epochs[name].append(global_score)
        if metric_scores[Cider().method()][0] > best_cider:
            best_cider = metric_scores[Cider().method()][0]
            print('New best CIDEr score is', best_cider)
            print('Saving model at', ckpt_path)
            save_checkpoint(model, ckpt_path)

        current_bleu4 = metric_scores[Bleu(4).method()][0]
        if current_bleu4 > best_bleu4:
            best_bleu4 = current_bleu4
            print('New best BLEU@4 score is', best_bleu4, flush=True)

        # Print some random captions to see that the model is learning something
        print('Some random captions from the model for the val set:', flush=True)
        for caption in random.sample(generated_captions, 5):
            print(caption, flush=True)

    print(f'Training complete. Best CIDEr: {best_cider:.4f} | Best BLEU@4: {best_bleu4:.4f}', flush=True)

    # Plot losses and metric scores
    Path('plots').mkdir(parents=True, exist_ok=True)
    plot_loss(f"plots/loss_plot_{_user}_{current_time}.jpg", step_losses, epoch_start_steps)
    plot_metrics(f"plots/metrics_plot_{_user}_{current_time}.jpg", metric_scores_epochs, config.num_epochs)


def save_checkpoint(model, ckpt_path: Path):
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)
    checkpoint = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path)


def debug():
    feature_dim = 20
    embed_size = 25
    hidden_size = 51
    vocab_size = 5

    # Instantiate the model, loss function, and optimizer.
    model = ImageCaptionModel(feature_dim, embed_size, hidden_size, vocab_size, 7, 2, 'LSTM')
    features = torch.randn(3, feature_dim, 3, 3)
    captions = torch.randint(0, 5, (3, 7))
    print(model(features, captions))


if __name__ == '__main__':
    # debug()
    train()
