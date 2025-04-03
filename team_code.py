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
# from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from scipy.signal import medfilt
import bottleneck as bn

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
        

        ecg = bn.move_median(ecg, window=12, min_count=1)

        
        
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

# # 2. Arquitectura del modelo
# class ChagasClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Bloque CNN
#         self.cnn = nn.Sequential(
#             nn.Conv1d(12, 64, 15, padding=7),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
            
#             nn.Conv1d(64, 128, 7, padding=3),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
            
#             nn.Conv1d(128, 256, 5, padding=2),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)  # [batch, 256, 1]
#         )
        
#         # Bloque Transformer
#         self.transformer = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        
#         # Clasificación
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         # CNN
#         cnn_features = self.cnn(x).squeeze(-1)  # [batch, 256]
        
#         # Transformer
#         transformer_in = cnn_features.unsqueeze(1)  # [batch, 1, 256]
#         attn_output, _ = self.transformer(transformer_in, transformer_in, transformer_in)
#         attn_output = attn_output.mean(dim=1)  # [batch, 256]
        
#         # Concatenación
#         combined = torch.cat([cnn_features, attn_output], dim=1)
        
#         # Clasificación
#         return self.classifier(combined)
    


# # 3. Función de entrenamiento y evaluación
# def train_and_save_model(X, y, model_folder):

#     # Configuración inicial
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Split de datos
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
#     # Crear datasets
#     train_dataset = ECGDataset(X_train, y_train)
#     val_dataset = ECGDataset(X_val, y_val, augment=False)
    
    
#     # DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32)
    
    
#     # Inicializar modelo
#     model = ChagasClassifier().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = nn.BCEWithLogitsLoss()
    
#     # Entrenamiento
#     best_val_acc = 0
#     for epoch in range(50):
#         print(epoch)
#         # Modo entrenamiento
#         model.train()
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs).squeeze()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
        
#         # Validación
#         model.eval()
#         all_preds = []
#         all_labels = []
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs).squeeze()
#                 preds = torch.sigmoid(outputs) > 0.5
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         # Métricas
#         val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
#         print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
#         print(classification_report(all_labels, all_preds, target_names=['No Chagas', 'Chagas']))
        
#         # Early stopping
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))




class ChagasClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)  # [batch, 256, 1]
        transformer_in = cnn_features.permute(2, 0, 1)  # [1, batch, 256]
        attn_output = self.transformer(transformer_in).mean(dim=0)  # [batch, 256]
        combined = torch.cat([cnn_features.squeeze(-1), attn_output], dim=1)
        return self.classifier(combined)

def train_and_save_model(X, y, model_folder,neg_count,pos_count):
    # Configuración inicial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split de datos
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Crear datasets
    train_dataset = ECGDataset(X_train, y_train, augment=False)
    val_dataset = ECGDataset(X_val, y_val, augment=False)
    
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    
    model = ChagasClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    neg_count = neg_count
    pos_count = pos_count
    pos_weight = torch.tensor([neg_count / pos_count], device=device)  
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_f1 = f1_score(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        scheduler.step(val_auc)

#############################################################################################################################################


# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.



# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    # records = find_records(data_folder, '.hea') # Not needed if obtain_balanced_train_dataset() used

    records = obtain_balanced_train_dataset(data_folder)
    
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    labels = np.zeros(num_records, dtype=bool)
    signals = []

    neg_count = 0
    pos_count = 0

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        
        
        # record = os.path.join(data_folder, records[i]) # Not needed if obtain_balanced_train_dataset() used
        
        record = records[i]

        labels[i] = load_label(record)

        if load_label(record) == 0:
            neg_count+=1
        if load_label(record) == 1:
            pos_count+=1

        signal_data = load_signals(record)

        signal = signal_data[0]

        signal = Zero_pad_leads(signal)
        signals.append(signal)

    print(neg_count)
    print(pos_count)
    # Train the models.
    if verbose:
        print('Training the model on the data...')

    signals = np.stack(signals, axis=0)
    
    print(labels.shape)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)



    train_and_save_model(signals,labels,model_folder,neg_count,pos_count)
    
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

    label = np.zeros(1, dtype=bool)

    signal_data = load_signals(record)
        
    signal = signal_data[0]
    signal = Zero_pad_leads(signal)
    signal = np.stack([signal], axis=0)


    test_dataset = ECGDataset(signal,label,augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Get the model outputs.
    model.eval()
    binary_output = []
    probability_output = [] 
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs= inputs.to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)
            
            probability_output = probs.item()
            print(probability_output)

            binary_output = probs > 0.5
            
            
    return binary_output, probability_output


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

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


# def obtain_balanced_train_dataset(path):
#         records = find_records(path, '.hea')

#         for i in range(0,len(records)):
#             records[i] = os.path.join(path, records[i])
            
#         # Obtener registros positivos con su edad y sexo
#         positive_records = []
#         age_sex_distribution = []
#         for rec in records:
#             if load_label(rec) == 1:
#                 head = load_header(rec)
#                 age = get_age(head)
#                 sex = get_sex(head)
#                 positive_records.append(rec)
#                 age_sex_distribution.append((age, sex))
        
#         num_positives = len(positive_records)
#         if num_positives == 0:
#             raise ValueError("No hay registros positivos en la base de datos")
        
       
        
#         positive_distribution = Counter((age_group(age), sex) for age, sex in age_sex_distribution)
        
#         # Obtener registros negativos
#         negative_candidates = [rec for rec in records if load_label(rec) == 0]
#         random.shuffle(negative_candidates)  # Barajar los negativos antes de seleccionarlos
        
#         selected_negatives = []
#         negative_distribution = Counter()
        
#         for rec in negative_candidates:
#             if len(selected_negatives) >= num_positives:
#                 break
            
#             head = load_header(rec)
#             age = get_age(head)
#             sex = get_sex(head)
#             age_bin = age_group(age)
            
#             if negative_distribution[(age_bin, sex)] < positive_distribution[(age_bin, sex)]:
#                 selected_negatives.append(rec)
#                 negative_distribution[(age_bin, sex)] += 1

#         return positive_records + selected_negatives

# import pandas as pd
# from sklearn.model_selection import train_test_split

def obtain_balanced_train_dataset(data_folder):
    records = find_records(data_folder, '.hea')

    for i in range(0, len(records)):
        records[i] = os.path.join(data_folder, records[i])
        
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
    
    # Definir rango flexible de negativos (entre el mismo número y un 50% más)
    min_negatives = num_positives
    max_negatives = int(num_positives * 1.5)
    
    positive_distribution = Counter((age_group(age), sex) for age, sex in age_sex_distribution)
    
    # Obtener registros negativos
    negative_candidates = [rec for rec in records if load_label(rec) == 0]
    random.shuffle(negative_candidates)  # Barajar los negativos antes de seleccionarlos
    
    selected_negatives = []
    negative_distribution = Counter()
    threshold_factor = 2  # Factor inicial de flexibilidad en la distribución
    
    while len(selected_negatives) < min_negatives and threshold_factor <= 2.5:
        remaining_candidates = [rec for rec in negative_candidates if rec not in selected_negatives]
        if not remaining_candidates:
            break  # Si ya hemos recorrido todos los registros, salimos
        
        for rec in remaining_candidates:
            if len(selected_negatives) >= max_negatives:
                break
            
            head = load_header(rec)
            age = get_age(head)
            sex = get_sex(head)
            age_bin = age_group(age)
            
            # Permitir cierta flexibilidad en la distribución
            if negative_distribution[(age_bin, sex)] < positive_distribution[(age_bin, sex)] * threshold_factor:
                selected_negatives.append(rec)
                negative_distribution[(age_bin, sex)] += 1
        
        # Aumentar el umbral de flexibilidad si no se han conseguido suficientes negativos
        if len(selected_negatives) < min_negatives:
            threshold_factor += 0.1
        
    print(len(positive_records))
    print(len(selected_negatives))

    return positive_records + selected_negatives


