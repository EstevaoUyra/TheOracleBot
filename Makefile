install:
	pip install -r requirements.txt
	python src/download_bert.py
	python gpt/download_model.py 345M

run:
	python src/start_bot.py