class Config:
    def __init__(self):
        # Captions and ResNet50 features file locations
        # Server path: '/itf-fi-ml/shared/courses/IN3310/mandatory2_data'
        import os
        self._root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mandatory2_data')
        self.train_caption_file = f'{self._root_dir}/annotations/captions_train2017.json'
        self.val_caption_file = f'{self._root_dir}/annotations/captions_val2017.json'
        self.val_images_dir = f'{self._root_dir}/val2017'
        self.resnet50_features_train_file = f'{self._root_dir}/coco_train_resnet18_layer4_features.pkl'
        self.resnet50_features_val_file = f'{self._root_dir}/coco_val_resnet18_layer4_features.pkl'

        # Model architecture
        self.max_caption_length = 30
        self.embedding_size = 512
        self.hidden_size = 512
        self.use_attention = False
        self.feature_size = 512
        self.num_layers = 2
        self.cell_type = 'RNN'  # 'LSTM'

        # Vocabulary
        self.vocabulary_file = f'{self._root_dir}/vocabulary.csv'
        self.vocabulary_size = 5000

        # Optimisation
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 5
        self.batch_size = 512
