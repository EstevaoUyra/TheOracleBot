from deeppavlov.core.data.utils import download
import os

if not os.path.exists('data/models/glove.txt'):
    download('data/models/glove.txt', source_url='http://files.deeppavlov.ai/embeddings/glove.6B.100d.txt')

