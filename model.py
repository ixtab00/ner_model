import transform_data
import load_data
import time

dl = load_data.DataLoader()
dt = transform_data.DataTransformer()

dataset, entities, words = dl.load_dataset('./dataset/train.txt')
embedding = dl.load_embedding('./embeddings/embedding.txt')
print(f"started: {time.ctime(time.time())}")
vocab = dt.get_vocab(words)
dataset = dt.transform_words(dataset, vocab)
embedding = dt.transform_embedding(embedding, vocab)
print(f"finished: {time.ctime(time.time())}")
