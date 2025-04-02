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

from helper_code import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import random
from collections import Counter

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# 1. Clase Dataset con preprocesamiento
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

# 2. Arquitectura del modelo
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
    


# 3. Función de entrenamiento y evaluación
def train_and_save_model(X, y, model_folder):

    # Configuración inicial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split de datos
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Crear datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val, augment=False)
    
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    
    # Inicializar modelo
    model = ChagasClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Entrenamiento
    best_val_acc = 0
    for epoch in range(50):
        print(epoch)
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
            torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))
    

#############################################################################################################################################


# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    
    records = obtain_balanced_train_dataset(data_folder)
    
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    labels = np.zeros(num_records, dtype=bool)
    signals = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        # record = os.path.join(data_folder, records[i])
        record = records[i]

        labels[i] = load_label(record)

        signal_data = load_signals(record)

        signal = signal_data[0]

        signal = Zero_pad_leads(signal)
        signals.append(signal)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    signals = np.stack(signals, axis=0)
    
    print(labels.shape)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    train_and_save_model(signals,labels,model_folder)
    
    if verbose:
        print('Done.')
        print()



# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChagasClassifier().to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pth')))

    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    
    print(record)

    label = load_label(record)

    signal_data = load_signals(record)
        
    signal = signal_data[0]
    signal = Zero_pad_leads(signal)

    test_dataset = ECGDataset(signal,label,augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Get the model outputs.
    model.eval()
    test_preds = []
    test_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = torch.sigmoid(outputs) > 0.5
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    return test_labels, test_preds

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

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




 # Agrupar edades en intervalos de 5 años
def age_group(age):
    return (age // 5) * 5


def obtain_balanced_train_dataset(path):
        records = find_records(path, '.hea')

        for i in range(0,len(records)):
            records[i] = os.path.join(path, records[i])

        
        
        # Obtener registros positivos con su edad y sexo
        positive_records = []
        age_sex_distribution = []
        for rec in records:
            if load_label(rec) == 1:
                head = load_header(rec)
                age = get_age(head)
                sex = get_sex(head)
                positive_records.append(rec)
                age_sex_distribution.append((age, sex))
        
        num_positives = len(positive_records)
        if num_positives == 0:
            raise ValueError("No hay registros positivos en la base de datos")
        
       
        
        positive_distribution = Counter((age_group(age), sex) for age, sex in age_sex_distribution)
        
        # Obtener registros negativos
        negative_candidates = [rec for rec in records if load_label(rec) == 0]
        random.shuffle(negative_candidates)  # Barajar los negativos antes de seleccionarlos
        
        selected_negatives = []
        negative_distribution = Counter()
        
        for rec in negative_candidates:
            if len(selected_negatives) >= num_positives:
                break
            
            head = load_header(rec)
            age = get_age(head)
            sex = get_sex(head)
            age_bin = age_group(age)
            
            if negative_distribution[(age_bin, sex)] < positive_distribution[(age_bin, sex)]:
                selected_negatives.append(rec)
                negative_distribution[(age_bin, sex)] += 1

        return positive_records + selected_negatives

