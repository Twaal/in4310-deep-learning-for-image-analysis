import torch


@torch.no_grad()
def evaluate_model(val_loader, model, device, metrics):
    model.eval()
    all_gts = {}  # image_id -> list of tokenized reference captions (list of lists of str)
    all_res = {}  # image_id -> list containing the generated caption (list with one str)
    generated_captions = []

    vocab = val_loader.dataset.vocab
    # Assume the start token is in your vocab (e.g. "<start>") and you use its index.
    start_token = vocab.word2idx['<start>']

    for features, captions, image_ids in val_loader:
        features = features.to(device)
        batch_size = features.size(0)
        # Create a dummy input: a column of start tokens.
        # This tensor is used to kick off the decoding.
        input_tokens = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Run the model in inference mode.
        # It should generate outputs for a fixed number of time steps (max_len).
        # The forward method should use argmax decoding when is_train=False.
        logits, alphas = model(features, input_tokens, is_train=False)
        # logits: [batch, seq_len, vocabulary_size]
        predicted_idxs = torch.argmax(logits, dim=-1)  # shape: [batch, seq_len]

        for i, img_id in enumerate(image_ids):
            # Convert predicted indices into a list of words.
            pred_idxs = predicted_idxs[i].tolist()
            generated_caption = vocab.get_sentence(pred_idxs)
            # Store the generated caption (wrapped as a single-item list)
            all_res[img_id] = [generated_caption]
            generated_captions.append(all_res[img_id])
            all_gts[img_id] = captions[i]  # Store the ground truth captions
            # For the ground-truth captions, TOKENIZE each caption.
            # tokenized_refs = [' '.join(word_tokenize(ref)) for ref in captions[i]]
            # all_gts[img_id] = tokenized_refs

    # Now compute each metric score.
    metric_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(all_gts, all_res)
        metric_name = metric.method()  # e.g., "CIDEr" or "BLEU"
        metric_scores[metric_name] = (score, scores)

    return metric_scores, generated_captions
