.PHONY: setup fetch preprocess features train evaluate all clean test

setup:
	pip install -r requirements.txt
	python setup.py install
	mkdir -p data/raw data/processed data/features data/splits
	mkdir -p models results/metrics results/plots results/reports logs

fetch:
	python scripts/fetch_data.py

preprocess:
	python scripts/preprocess_data.py

features:
	python scripts/engineer_features.py

train:
	python scripts/train_all_models.py

evaluate:
	python scripts/evaluate_models.py

all: fetch preprocess features train evaluate

clean:
	rm -rf data/raw/* data/processed/* data/features/* data/splits/*
	rm -rf models/* results/metrics/* results/plots/* results/reports/*
	rm -rf logs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/ -v --cov=src
