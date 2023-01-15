from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from tqdm import tqdm
from data import GenreDataset

model = AutoModelForSequenceClassification.from_pretrained(
    './model/results/checkpoint-20000/')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_prediction(text):
    encoding = tokenizer(text, return_tensors="pt",
                         padding=True, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    outputs = model(**encoding).logits

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.squeeze().cpu())
    probs = probs.detach().numpy()
    label = np.argmax(probs, axis=-1)
    #  Get the labels that has probability > 0.85
    best_labels = np.argwhere(probs > 0.85).squeeze().tolist()
    return {
        'main_genre': genre_map_inv[label],
        'genres': best_labels,
        # dict(zip(genre_map_inv.values(), probs)),
        'probs': dict(zip(genre_map_inv.values(), probs)),
        'best_labels': [genre_map_inv[i] for i in best_labels] if best_labels else None,
    }


def get_batch_prediction(dataset):
    preds = []
    for batch in tqdm(dataset):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch).logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(outputs.squeeze().cpu())
        probs = probs.detach().numpy()
        label = np.argmax(probs, axis=-1)
        #  Get the labels that has probability > 0.5
        best_labels = np.argwhere(probs > 0.5).squeeze().tolist()
        preds.append({
            'label': label,
            'probs': probs,
            'best_labels': best_labels,
            'best_labels_names': [genre_map_inv[i] for i in best_labels] if best_labels else None,
        })
    return preds


if __name__ == '__main__':
    #  Path to the test data
    TEST_PATH = './data/Genre Classification Dataset/test_data_solution.txt'
    # ID ::: TITLE ::: GENRE ::: DESCRIPTION
    TRAIN_PATH = './data/Genre Classification Dataset/train_data.txt'

    # Read the data
    train = pd.read_csv(TRAIN_PATH, sep=':::', names=[
        'id', 'title', 'genre', 'description'], engine='python')
    test = pd.read_csv(TEST_PATH, sep=':::', names=[
        'id', 'title', 'genre', 'description'], engine='python')

    #  Encode the labels and save the mapping
    genre_map = {genre: i for i, genre in enumerate(train['genre'].unique())}
    train['genre'] = train['genre'].factorize()[0].astype('int')
    test['genre'] = test['genre'].factorize()[0].astype('int')
    genre_map_inv = {v: k for k, v in genre_map.items()}

    test_encodings = tokenizer(
        test['description'].tolist(), truncation=True, padding=True)
    test_dataset = GenreDataset(test_encodings, test['genre'].tolist())
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False)

    ind = int(np.random.random()*test.shape[0])
    print(ind)
    print(test.description[ind])
    import time
    start = time.time()
    print(get_prediction(test.description[ind]))
    print(f'Time taken in seconds: {time.time()-start}')
    print(f'Ground truth: {genre_map_inv[test.genre[ind]]}')
