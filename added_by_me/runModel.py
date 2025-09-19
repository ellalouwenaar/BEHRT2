from preprocess_dataframe import preprocess_to_behrt_format, create_vocabularies
from dataLoaderBERHT import BEHRTDataset
from Create_BEHRT_embedding import BEHRT
import torch       
from extractEmbedding import extract_patient_embeddings
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_pickle(r"C:\Users\E.Louwenaar\Documents\Thesis2\cleaned_mincount50_medL4 copy.pkl")

# 1. Preprocess data
behrt_df = preprocess_to_behrt_format(df)

# 2. Maak vocabularies
vocab_dict, age2idx = create_vocabularies(behrt_df)

print(f"Vocab size: {len(vocab_dict['token2idx'])}, Age vocab size: {len(age2idx)}")
print(f"Voorbeeld tokens: {list(vocab_dict['token2idx'].items())[:10]}")
print(f"Voorbeeld ages: {list(age2idx.items())[:10]}")

# 3. Maak dataset en dataloader
dataset = BEHRTDataset(behrt_df, vocab_dict['token2idx'], age2idx)

# training and test data together
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

print(dataset[0])
print(len(dataset))
print(dataset[1])

# 4. Initialiseer model
model = BEHRT(
    vocab_size=len(vocab_dict['token2idx']), 
    age_vocab_size=len(age2idx)
)

# # 5. Laad pre-trained weights indien beschikbaar
# # model.load_state_dict(torch.load('behrt_pretrained.pt'))

# 6. Extract embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
patient_embeddings = extract_patient_embeddings(model, dataloader, device)


embedding_matrix = np.array(list(patient_embeddings.values()))
patient_ids = list(patient_embeddings.keys())

print(f"Extracted embeddings for {len(patient_embeddings)} patients.")
print(f"Embedding shape: {embedding_matrix.shape}")
print(f"Voorbeeld patient ID's: {patient_ids[:2]}")
print(f"Voorbeeld embeddings: {embedding_matrix[:2]}")

# Cluster patiënten
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embedding_matrix)