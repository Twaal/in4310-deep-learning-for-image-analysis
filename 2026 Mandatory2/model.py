import torch
from torch import nn


class ImageCaptionModel(nn.Module):
    def __init__(self, cnn_feature_dim, embed_size, hidden_size, vocab_size, max_caption_length, num_layers, cell_type,
                 use_attention=False):
        """
        The main Image Captioning Model class.
        :param cnn_feature_dim (int): Dimensionality of the frozen image features from ResNet.
        :param embed_size (int): Dimensionality of the word embeddings.
        :param hidden_size (int): Hidden state size of the RNN/LSTM cell.
        :param vocab_size (int): Size of the vocabulary.
        :param max_caption_length (int): Maximum caption length.
        :param num_layers (int): Number of layers of RNN/LSTM cell.
        :param cell_type (str): 'RNN' or 'LSTM'.
        :param use_attention (bool): Use the attention mechanism if True
        """
        super(ImageCaptionModel, self).__init__()
        # TODO: Check the task description and replace None with the correct feature projection layer
        self.feature_projection = None
        # TODO: Implement Global Average Pooling using torch.nn.AdaptiveAvgPool2d
        self.avgpool = None
        self.caption_rnn = CaptionRNN(
            cnn_feature_dim=cnn_feature_dim,
            vocabulary_size=vocab_size,
            embedding_size=embed_size,
            hidden_state_size=hidden_size,
            max_caption_length=max_caption_length,
            num_layers=num_layers,
            cell_type=cell_type,
            use_attention=use_attention
        )

    def forward(self, cnn_features, token_ids, is_train):
        """
        :param cnn_features (torch.Tensor): Features from the CNN; shape [batch, num_regions, feat_dim].
        :param token_ids (torch.Tensor): Token indices, shape [batch, seq_len]. The 1st token should be the start token.
        :param is_train (bool): If True, use teacher forcing; otherwise, use inference mode.
        :return: logits (torch.Tensor): Logits for each time step. Shape: [seq_len, batch, vocabulary_size].
                attn_weights (List): Returned only if use_attention is True. A list of attention map weights for each
                                     time step. Shape: [seq_len, batch, num_regions]
        """
        if self.use_attention:
            # TODO: Permute features: from [batch, channels, H, W] to [batch, H, W, channels],
            #       then flatten spatial dimensions to get [batch, num_regions, channels].
            cnn_features = None
            # TODO: Apply the projection to each region (nn.Linear applies to the last dim).
            processed_img_feat = None  # Resulting shape should be [batch, num_regions, hidden_size]
            logits, attn_weights = self.caption_rnn(
                tokens=token_ids,
                features=processed_img_feat,
                is_train=is_train
            )
            return logits, attn_weights
        else:
            # TODO: Use self.avgpool to convert features from shape [batch_size, 512, 7, 7] to [batch_size, 512]
            cnn_features = None
            processed_img_feat = self.feature_projection(cnn_features)  # Resulting shape: [batch_size, hidden_size]
            logits, final_hidden = self.caption_rnn(
                tokens=token_ids,
                features=processed_img_feat,
                is_train=is_train
            )
            return logits, None


class CaptionRNN(nn.Module):
    def __init__(self, cnn_feature_dim, vocabulary_size, embedding_size,
                 hidden_state_size, max_caption_length, num_layers, cell_type, use_attention=False):
        """
        Combined RNN with optional attention mechanism.

        Args:
            cnn_feature_dim (int): Dimensionality of the (possibly projected) feature vector(s).
            vocabulary_size (int): Vocabulary size.
            embedding_size (int): Dimensionality of token embeddings.
            hidden_state_size (int): Hidden state size of the RNN.
            max_caption_length (int): Maximum caption length.
            num_layers (int): Number of RNN layers.
            cell_type (str): 'RNN' or 'LSTM'.
            use_attention (bool): If True, use attention mechanism over features.
        """
        super(CaptionRNN, self).__init__()
        self.use_attention = use_attention
        self.hidden_state_size = hidden_state_size
        self.num_layers = num_layers
        self.max_caption_length = max_caption_length
        self.vocabulary_size = vocabulary_size

        # Embedding and output layers.
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # TODO: The output layer (final layer) is a linear layer. What should be the size (dimensions) of its output?
        #         Replace None with a linear layer with correct output size
        self.output_layer = None  # nn.Linear(self.hidden_state_size, )

        # Create attention module if needed.
        if self.use_attention:
            self.attention = Attention(cnn_feature_dim=cnn_feature_dim, hidden_dim=hidden_state_size)

        # TODO: len(input_size_list) == num_rnn_layers and input_size_list[i] should contain the input size for layer i.
        # This is used to populate self.cells
        # For the first layer, the input is the concatenation of the token embedding and
        # the raw feature vector or the attention-weighted feature vector.
        input_sizes = None

        # TODO: Create a list of type "nn.ModuleList" and populate it with cells (layers) of type self.cell_type.
        self.cells = None

    def forward(self, tokens, features, is_train):
        """
        Forward pass for the caption RNN.

        Args:
            tokens (Tensor): Token indices, shape [batch, seq_len].
            features (Tensor):
                If use_attention is True, expected shape [batch, num_regions, feature_dim];
                Otherwise, shape [batch, feature_dim].
            is_train (bool): If True, use teacher forcing.

        Returns:
            logits (Tensor): Logits for each time step, shape [seq_len, batch, vocabulary_size].
            attn_weights (list[Tensor] or None): List of attention weights per time step if using attention;
                                                 otherwise, None. Shape: [seq_len, batch, num_regions]
        """
        batch_size, seq_len = tokens.size()
        if not is_train:
            seq_len = self.max_caption_length

        # Embed all tokens (for teacher forcing).
        token_embeddings = self.embedding(tokens)  # [batch, seq_len, embedding_size]

        # TODO: Initialize hidden_states with correct dimensions depending on the cell type.
        # hidden_states is a list of length self.num_layers with each element having a tensor of zeros of shape
        # (batch_size, 2 * self.hidden_state_size).
        # We use (2 * self.hidden_state_size) because we need a hidden state AND a cell memory for LSTM.
        # We do not need this size for vanilla RNN cell but we modify the RNNCell instead to have the same
        # interface for both RNNCell and LSTMCell. This avoids putting if statements at some places in the
        # code below.
        hidden_states = None

        logits_series = []
        attn_weights_series = [] if self.use_attention else None

        # TODO: Fetch the first (index 0) embeddings that should go as input to the RNN.
        current_token_vec = None  # Should have shape (batch_size, embedding_size)

        for t in range(seq_len):
            new_states = []
            # TODO:
            # 1. Loop over the RNN layers and provide them with correct input. Inputs depend on the layer
            #    index so input for layer-0 will be different from the input for other layers.
            # 2. Update the hidden cell state for every layer.
            # 3. If you are at the last layer, then produce logits_i, predictions. Append logits_i to logits_series.
            for layer in range(self.num_layers):
                if layer == 0:
                    if self.use_attention:
                        # TODO:  Compute attention: use previous hidden state (only the first half of
                        #        hidden_states variable) to attend over features.
                        prev_hidden = None
                        context, alpha = self.attention(features, prev_hidden)
                        attn_weights_series.append(alpha)
                        # TODO: Concatenate token embedding with the attended image context.
                        cell_input = None
                    else:
                        # TODO: Without attention, concatenate the token embedding with the image feature.
                        cell_input = None
                else:
                    cell_input = None  # TODO: Initialise to the output of the previous layer

                new_state = None  # TODO: Call the cell for this layer
                new_states.append(new_state)

                if layer == self.num_layers - 1:
                    # TODO: Get logits from the self.output_layer and append them to logits_series
                    logits = None
                    logits_series.append(logits)
                    if t < seq_len - 1:
                        if is_train:
                            current_token_vec = token_embeddings[:, t + 1, :]
                        else:
                            predicted_tokens = torch.argmax(logits, dim=1)
                            current_token_vec = self.embedding(predicted_tokens)
            hidden_states = new_states

        logits = torch.stack(logits_series, dim=1)  # Convert to a tensor
        return logits, attn_weights_series


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_state_size):
        super(RNNCell, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_state_size)
        self.hidden_to_hidden = nn.Linear(hidden_state_size, hidden_state_size)

    def forward(self, x, state_old):
        # state_old: [batch, 2 * hidden_state_size]; we use only the first half.
        h_old = state_old[:, :state_old.size(1) // 2]
        h_new = torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h_old))
        # Concatenate new hidden state with itself to mimic (hidden_state, cell_memory) for LSTM-like interface.
        return torch.cat([h_new, h_new], dim=1)


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_state_size: int):
        """
        :param input_size: Size (number of units/features) of the input to LSTM
        :param hidden_state_size: Size (number of units/features) in the hidden state of LSTM
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        # TODO: Initialise weights and biases for the forget gate (weight_f, bias_f), input gate (w_i, b_i),
        #       output gate (w_o, b_o), and hidden state (weight, bias)
        #       self.weight, self.weight_(f, i, o):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(f, i, o): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to two.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n
        #       Note: The actual input tensor will have 2 * HIDDEN_STATE_SIZE because it contains both
        #             hidden state and cell's memory

        # Forget gate parameters
        self.weight_f = None
        self.bias_f = None
        # Input gate parameters
        self.weight_i = None
        self.bias_i = None
        # Output gate parameters
        self.weight_o = None
        self.bias_o = None
        # Memory cell parameters
        self.weight = None
        self.bias = None

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for an LSTM unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the LSTM.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the LSTM cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the LSTM equations to get the new hidden state, cell memory and return them.
        #       The first half of the returned value must represent the new hidden state and the second half
        #       new cell state.
        new_hidden_state = None
        return new_hidden_state


class Attention(nn.Module):
    def __init__(self, cnn_feature_dim, hidden_dim):
        """
        A simple attention module.
        Args:
            cnn_feature_dim (int): Number of channels in the encoder feature maps.
            hidden_dim (int): Dimensionality of the hidden state.
        """
        super(Attention, self).__init__()
        # TODO: Create layers to project the hidden state and the image features to a common dimension.
        self.hidden_to_hidden = None  # Linear layer to project hidden_state to hidden_dim
        self.features_to_hidden = None  # Linear layer to project cnn_feature_dim to hidden_dim

        self.hidden_to_attention_score = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_state):
        """
        Args:
            features (Tensor): Encoded image features,
                                  shape: [batch, num_regions, encoder_dim].
            hidden_state (Tensor): Hidden state from the LSTM,
                                   shape: [batch, hidden_dim].
        Returns:
            context (Tensor): Weighted image feature vector, shape: [batch, encoder_dim].
            alpha (Tensor): Attention weights, shape: [batch, num_regions].
        """
        # TODO: Project hidden state and unsqueeze() to [batch, 1, hidden_dim] so that it can be broadcast.
        hidden_proj = None
        # TODO: Project encoder outputs
        enc_proj = None  # Resulting Shape: [batch, num_regions, hidden_dim]
        # TODO: Add the projections and apply self.relu
        att = None
        # TODO: Get scalar attention scores for each region of the image
        att_scores = None  # Resulting shape: [batch, num_regions]
        # TODO: Apply softmax on the att_scores to get the alphas (attention weights)
        alpha = None
        # TODO: Compute context vector as the weighted sum of encoder outputs. You might need to unsqueeze() alpha.
        context = None
        return context, alpha
