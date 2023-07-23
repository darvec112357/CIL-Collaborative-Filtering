import numpy as np

import torch
import torch.nn as nn
from torch.nn import Linear, Embedding, Dropout, Flatten
from torch.utils.data import Dataset


# Dataset
class CFDataset(Dataset):
    def __init__(self, train_matrix):
        self.train_matrix = train_matrix

    def __len__(self):
        return len(self.train_matrix)

    def __getitem__(self, idx):
        return self.train_matrix[idx][0], self.train_matrix[idx][1], self.train_matrix[idx][2]

# Helper functions for training and testing
def nn_train(model, train_loader, loss_function, optimizer, num_epochs=10):
    model.train()
    running_loss = 0.0
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            user_indices, item_indices, ratings = data
            user_indices -= 1
            item_indices -= 1
            ratings = ratings.float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(user_indices, item_indices)
            loss = loss_function(outputs, ratings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss/100:.3f}", end='\r')
                running_loss = 0.0
    return model

def nn_predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for user_indices, item_indices, ratings in test_loader:
            user_indices -= 1
            item_indices -= 1
            ratings = ratings.float().unsqueeze(1)

            outputs = model(user_indices, item_indices)
            predictions.extend(outputs.squeeze(1).tolist())
    return predictions


# Model
class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF) model
    """
    def __init__(self, latent_dim, num_users, num_items):
        super().__init__()

        self.embedding_user_mf = Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.embedding_item_mf = Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        self.fc_mf = nn.Linear(in_features=latent_dim, out_features=1)

    def forward(self, user_indices, item_indices):
        flat = Flatten()
        user_embedding = self.embedding_user_mf(user_indices)
        item_embedding = self.embedding_item_mf(item_indices)
        
        user_embedding = flat(user_embedding)
        item_embedding = flat(item_embedding)

        element_product = torch.mul(user_embedding, item_embedding)

        rating = self.fc_mf(element_product)
        return rating
    
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model (with Embeddings)
    """
    def __init__(self, latent_dim, num_users, num_items, hidden_layers=[128]):
        super().__init__()

        self.embedding_user_mlp = Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.embedding_item_mlp = Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        layers = []
        input_size = latent_dim * 2
        for layer_size in hidden_layers:
            layers.append(Linear(in_features=input_size, out_features=layer_size))
            layers.append(nn.ReLU())
            layers.append(Dropout(p=0.5))
            input_size = layer_size
        self.fc_mlp = nn.Sequential(*layers)
        self.fc_final = nn.Linear(in_features=input_size, out_features=1)

    def forward(self, user_indices, item_indices):
        flat = Flatten()
        user_embedding = self.embedding_user_mlp(user_indices)
        item_embedding = self.embedding_item_mlp(item_indices)
        
        user_embedding = flat(user_embedding)
        item_embedding = flat(item_embedding)

        concat = torch.cat([user_embedding, item_embedding], dim=-1)

        intermediate = self.fc_mlp(concat)
        rating = self.fc_final(intermediate)
        return rating

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) model
    Combines GMF and MLP models using pre-trained weights and weights them using alpha
    """
    def __init__(self, latent_dim, num_users, num_items, hidden_layers=[], pretrained=False, alpha=0.5):
        super().__init__()

        # MF part
        self.embedding_user_mf = Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.embedding_item_mf = Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        # MLP part
        self.embedding_user_mlp = Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.embedding_item_mlp = Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        layers = []
        input_size = latent_dim * 2
        for layer_size in hidden_layers:
            layers.append(Linear(in_features=input_size, out_features=layer_size))
            layers.append(nn.ReLU())
            layers.append(Dropout(p=0.5))
            input_size = layer_size
        self.fc_mlp = nn.Sequential(*layers)

        # Concatenate MF and MLP parts
        self.affine_output = nn.Linear(in_features=latent_dim + input_size, out_features=1)

        self.pretrained = pretrained
        self.alpha = alpha
        
    
    def forward(self, user_indices, item_indices):
        flat = Flatten()

        # MF part
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        
        user_embedding_mf = flat(user_embedding_mf)
        item_embedding_mf = flat(item_embedding_mf)

        pred_mf = torch.mul(user_embedding_mf, item_embedding_mf)

        # MLP part
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        
        user_embedding_mlp = flat(user_embedding_mlp)
        item_embedding_mlp = flat(item_embedding_mlp)

        concat_mlp = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

        pred_mlp = self.fc_mlp(concat_mlp)

        if self.pretrained:
            # Concatenate MF and MLP parts using alpha
            concat = torch.cat([self.alpha * pred_mf, (1 - self.alpha) * pred_mlp], dim=-1)        
        else:
            concat = torch.cat([pred_mf, pred_mlp], dim=-1)

        rating = self.affine_output(concat)
        return rating