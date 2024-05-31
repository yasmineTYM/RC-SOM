# PhraseMap: Attention-based Keyphrases Recommendation for Information Seeking
This repository contains the implementation of the RC-SOM algorithm presented in our paper: [https://ieeexplore.ieee.org/document/9964397]("PhraseMap: Attention-based Keyphrases Recommendation for Information Seeking".)

## Overview
The Resource-Controlled Self-Organizing Map (RC-SOM) extends the traditional Self-Organizing Map (SOM) to achieve resource-controlled mapping, making it suitable for dimensionality reduction from high-dimensional space to 2D cells.

The SOM is an unsupervised machine learning technique that projects high-dimensional data onto a low-dimensional discrete map while preserving the data structure. 

## Usage 
Below is an example of how to call our functions:

```python
import numpy as np
from sklearn.manifold import TSNE
from minisom import MiniSom

def SOM(size_x, size_y, original_embedding, embedding_keys, sigma, lr, epoch):
    print('=========computing tsne=========')
    original_embedding = np.nan_to_num(original_embedding)  # Replace NaN with 0
    embedding = TSNE(n_components=2, init='random', random_state=23).fit_transform(original_embedding)
    print('=========train som=========')
    som = MiniSom(size_x, size_y, embedding.shape[1], sigma=sigma, resource_limit=1, learning_rate=lr,
                  activation_distance='euclidean', topology='hexagonal', neighborhood_function='gaussian', random_seed=10)
    som.train(embedding, epoch, 10, verbose=True)
    
    res = {}
    som._resource_initialize()
    for cnt, x in enumerate(embedding):
        # Getting the winner
        w = som.winner(x)
        if res.get(f"({w[0]},{w[1]})"):
            res[f"({w[0]},{w[1]})"].append(embedding_keys[cnt])
        else:
            res[f"({w[0]},{w[1]})"] = [embedding_keys[cnt]]
    
    return res  # Mapping from position -> key
```


```python
# Example embeddings and keys (replace with your actual data)
original_embedding = np.random.rand(100, 768)  # Example: 100 phrases with 768-dim embeddings
embedding_keys = [f"key_{i}" for i in range(100)]

# Parameters
size_x = 10
size_y = 10
sigma = 1.0
lr = 0.5
epoch = 1000

# Running the SOM function
mapping = SOM(size_x, size_y, original_embedding, embedding_keys, sigma, lr, epoch)
print(mapping)
```

## Description 
We implement RC-SOM based on the [https://pypi.org/project/MiniSom/](MiniSom). 
