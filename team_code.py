#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################


#################################################################################################################################
import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Configuración inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Generación de datos de ejemplo 
# X = dataset_signals
# y = dataset_labels

# 3. Clase Dataset con preprocesamiento
class ECGDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = X
        self.y = y
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ecg = self.X[idx]
        
        # Normalización por derivación
        ecg = (ecg - ecg.mean(axis=0)) / (ecg.std(axis=0) + 1e-8)
        
        # Aumento de datos
        if self.augment:
            # Desplazamiento temporal
            shift = np.random.randint(-100, 100)
            ecg = np.roll(ecg, shift, axis=0)
            
            # Ruido gaussiano
            if np.random.rand() > 0.7:
                ecg += np.random.normal(0, 0.01, ecg.shape)
                
        return torch.tensor(ecg.T, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# 4. Arquitectura del modelo
class ChagasClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Bloque CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # [batch, 256, 1]
        )
        
        # Bloque Transformer
        self.transformer = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        
        # Clasificación
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # CNN
        cnn_features = self.cnn(x).squeeze(-1)  # [batch, 256]
        
        # Transformer
        transformer_in = cnn_features.unsqueeze(1)  # [batch, 1, 256]
        attn_output, _ = self.transformer(transformer_in, transformer_in, transformer_in)
        attn_output = attn_output.mean(dim=1)  # [batch, 256]
        
        # Concatenación
        combined = torch.cat([cnn_features, attn_output], dim=1)
        
        # Clasificación
        return self.classifier(combined)
    


# 5. Función de entrenamiento y evaluación
def train_and_evaluate(X, y):
    # Split de datos
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Crear datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val, augment=False)
    test_dataset = ECGDataset(X_test, y_test, augment=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Inicializar modelo
    model = ChagasClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Entrenamiento
    best_val_acc = 0
    for epoch in range(50):
        # Modo entrenamiento
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validación
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Métricas
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=['No Chagas', 'Chagas']))
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Evaluación final
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print("\n--- Resultados Finales en Test ---")
    print(classification_report(test_labels, test_preds, target_names=['No Chagas', 'Chagas']))
    print("Matriz de Confusión:")
    print(confusion_matrix(test_labels, test_preds))
    
    return model
    

#############################################################################################################################################


# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 6), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)
    signals = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        labels[i] = load_label(record)

        signal_data = load_signals(record)
        if i == 0:
            print(record)
            print(signal_data)

        signal = signal_data[0]
        signal = Zero_pad_leads(signal)
        signals.append(signal)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    signals = np.stack(signals, axis=0)
    

    print(labels.shape)


    model = ChagasClassifier()



    # # This very simple model trains a random forest model with very simple features.

    # # Define the parameters for the random forest classifier and regressor.
    # n_estimators = 12  # Number of trees in the forest.
    # max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    # random_state = 56  # Random state; set for reproducibility.

    # # Fit the model.

    # model = RandomForestClassifier(
    #     n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    print(model)
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


def Zero_pad_leads(arr, target_length=4096):
    X, Y = arr.shape  # X filas, 12 columnas
    padded_array = np.zeros((target_length, Y))  # Matriz destino con ceros
    
    for col in range(Y):
        col_data = arr[:, col]  # Extraer columna
        length = len(col_data)
        
        if length < target_length:
            pad_before = (target_length - length) // 2
            pad_after = target_length - length - pad_before
            padded_array[:, col] = np.pad(col_data, (pad_before, pad_after), mode='constant')
        else:
            padded_array[:, col] = col_data[:target_length]  # Recortar si es más largo
    
    return padded_array