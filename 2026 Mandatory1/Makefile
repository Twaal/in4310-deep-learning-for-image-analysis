IMAGE_NAME = mandatory1
DATA_DIR = ./data

.PHONY: docker train

docker:
	docker build -t $(IMAGE_NAME) .

test_data:
	docker run --rm --name $(IMAGE_NAME) -v $(DATA_DIR):/app/data $(IMAGE_NAME) python src/dataset.py

test_model:
	docker run --rm --name $(IMAGE_NAME) -v $(DATA_DIR):/app/data -v ./models:/app/models $(IMAGE_NAME) python src/model.py

train:
	docker run --rm --name $(IMAGE_NAME) --gpus all --shm-size=2g -v $(DATA_DIR):/app/data -v ./models:/app/models $(IMAGE_NAME) python src/train.py

evaluate:
	docker run --rm --name $(IMAGE_NAME) --gpus all --shm-size=2g -v $(DATA_DIR):/app/data -v ./models:/app/models $(IMAGE_NAME) python src/evaluate.py

feature_maps:
	docker run --rm --name $(IMAGE_NAME) --gpus all --shm-size=2g -v $(DATA_DIR):/app/data -v ./models:/app/models -v ./report:/app/report $(IMAGE_NAME) python src/feature_maps_statistics.py
