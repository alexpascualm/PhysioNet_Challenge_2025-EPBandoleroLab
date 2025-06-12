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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from scipy.signal import medfilt, butter, filtfilt

import numpy as np
# from sklearn.metrics import classification_report

from collections import Counter, defaultdict
import math
import random
# import pickle


PROB_THRESHOLD = 0.6

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# 1. Clase Dataset con preprocesamiento
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ecg = self.X[idx]

        # Normalización por derivación
        ecg = (ecg - ecg.mean(axis=0)) / (ecg.std(axis=0) + 1e-8) 

        return torch.tensor(ecg.T, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# 2. Arquitectura del modelo
class ChagasClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 12, 10, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(12, 64, 10, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 10, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 10, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, 10, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10)  # Secuencia de longitud 10
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024),
            num_layers=3
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)  # [batch, 512, 10]
        transformer_in = cnn_features.permute(0, 2, 1)  # [batch, 10, 512]
        attn_output = self.transformer(transformer_in).mean(dim=1)  # [batch, 512]
        combined = torch.cat([cnn_features.mean(dim=2), attn_output], dim=1)  # [batch, 1024]
        return self.classifier(combined)

    
class FocalLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.5, gamma=2.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        # Inicializar BCEWithLogitsLoss con pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

    def forward(self, inputs, targets):
        # Calcular la pérdida BCE con pos_weight
        BCE_loss = self.bce_loss(inputs, targets)
        # Calcular pt (probabilidad de la clase correcta)
        pt = torch.exp(-BCE_loss)
        # Aplicar Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()



# 3. Función de entrenamiento y guardado
def train_and_save_model(X, y, model_folder,obtain_test_metrics):
    # Configuración inicial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("GPU Available")
    else:
        print("GPU not available")

    # Split de datos
    if obtain_test_metrics:
        X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_aux, y_aux, test_size=0.5, stratify=y_aux, random_state=42)

        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)   
    
    # Crear datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
        
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    
    
    model = ChagasClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / pos_count

    pos_weight = torch.tensor([pos_weight], device=device)  
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss(pos_weight, alpha=0.7, gamma=1.0)
    
    best_challenge_score = 0
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(50):
        # Entrenamiento
        model.train()
        train_loss = 0
        train_probs, train_labels = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Guardar probabilidades para métricas
            probs = torch.sigmoid(outputs)
            train_probs.extend(probs.cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())
        
        train_loss = train_loss/len(train_loader)
        
        # Calcular métricas en el conjunto de entrenamiento
        train_pred_labels = np.array(train_probs) > PROB_THRESHOLD
        train_challenge_score = compute_challenge_score(train_labels, train_probs)
        train_auc, train_auprc = compute_auc(train_labels, train_probs)
        train_accuracy = compute_accuracy(train_labels, train_pred_labels)
        train_f1 = compute_f_measure(train_labels, train_pred_labels)
        
        # Validación
        model.eval()
        val_loss = 0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss/len(val_loader)

        # Calcular métricas en el conjunto de validación
        val_pred_labels = np.array(val_probs) > PROB_THRESHOLD
        val_challenge_score = compute_challenge_score(val_labels, val_probs)
        val_auc, val_auprc = compute_auc(val_labels, val_probs)
        val_accuracy = compute_accuracy(val_labels, val_pred_labels)
        val_f1 = compute_f_measure(val_labels, val_pred_labels)
        
       # Imprimir métricas
        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {train_loss:.4f}, Challenge Score: {train_challenge_score:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Challenge Score: {val_challenge_score:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")
        
        # Guardar el mejor modelo basado en el challenge_score de validación
        if val_challenge_score > best_challenge_score:
            best_challenge_score = val_challenge_score
            torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        
        # Ajustar el learning rate basado en el challenge_score de validación
        scheduler.step(val_challenge_score)
        
        # Detección de overfitting
        if train_challenge_score - val_challenge_score > 0.1:  # Umbral arbitrario para detectar overfitting
            print("Warning: Potential overfitting detected (Train Challenge Score significantly higher than Val)")

    if obtain_test_metrics:
        # Test
        model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pth'),weights_only=True))
        model.eval()
        test_loss = 0
        test_probs, test_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                    
                probs = torch.sigmoid(outputs)
                test_probs.extend(probs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
            
        test_loss = test_loss/len(test_loader)

        # Evaluate the model outputs.
        test_pred_labels = np.array(test_probs) > PROB_THRESHOLD
        test_challenge_score = compute_challenge_score(test_labels, test_probs)
        test_auc, test_auprc = compute_auc(test_labels, test_probs)
        test_accuracy = compute_accuracy(test_labels, test_pred_labels)
        test_f1 = compute_f_measure(test_labels, test_pred_labels)

                
        # Imprimir métricas
        print("Test inference")
        print(f"  Test - Loss: {test_loss:.4f}, Challenge Score: {test_challenge_score:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f},  AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}")
    
#############################################################################################################################################


# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    # records = find_records(data_folder, '.hea') # Not needed if obtain_balanced_train_dataset() used

    records = obtain_balanced_train_dataset(data_folder, negative_to_positive_ratio=1)
    
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    labels = np.zeros(num_records, dtype=bool)
    signals = []
    sources = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        
        # record = os.path.join(data_folder, records[i]) # Not needed if obtain_balanced_train_dataset() used
        record = records[i]

        labels[i] = load_label(record)

        signal_data = load_signals(record)
        source = load_source(record)

        signal = signal_data[0]

        signal = preprocess_12_lead_signal(signal)
       
        print("Signal shape:", signal.shape)
        
        signals.append(signal)
        sources.append(source)

 
    # Train the models.
    if verbose:
        print('Training the model on the data...')

    signals = np.stack(signals, axis=0)

    # signals = np.array(signals, dtype=object)
    # print("Sigue Furulando")
    # print(signals.shape)
    # print("Shape de signals[0]:", signals[0].shape)
    # print("Shape de signals[1]:", signals[1].shape)
    # print(labels.shape)

    # # Paquete con ambos objetos en orden
    # data = {
    #     'signals': signals,
    #     'labels': labels,
    #     'sources': sources
    # }

    # # Guardar en archivo pickle
    # with open('/home/jamon/alejandropm/PhysioNet_Challenge/PhysioNet_Challenge_2025-EPBandoleroLab/Challenge_Data.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    

    # # Apply data augmentation
    # if verbose:
    #     print('Applying data augmentation...')
    # n_augmentations = 3  # Number of augmented samples per original sample
    # signals, labels = augment_ecg_data(signals, labels, n_augmentations=n_augmentations)



    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    train_and_save_model(signals,labels,model_folder,obtain_test_metrics=True)
    
    if verbose:
        print('Done.')
        print()



# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChagasClassifier().to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pth'),weights_only=True))

    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    label = np.zeros(1, dtype=bool)

    signal_data = load_signals(record)
    signal = signal_data[0]
    signal = preprocess_12_lead_signal(signal)
    
    signal = np.stack([signal], axis=0)

    test_dataset = ECGDataset(signal,label)
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
            binary_output = probs > PROB_THRESHOLD
            
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


# def apply_median_filter(signal, fs=400, short_window_ms=200, long_window_ms=600):
#     short_window = int(fs * short_window_ms / 1000)
#     long_window = int(fs * long_window_ms / 1000)
#     if short_window % 2 == 0:
#         short_window += 1
#     if long_window % 2 == 0:
#         long_window += 1
#     baseline = medfilt(signal, kernel_size=short_window)
#     baseline = medfilt(baseline, kernel_size=long_window)
#     return signal - baseline

# def bandpass_filter(signal, fs=400, lowcut=0.5, highcut=30, order=4):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     filtered = filtfilt(b, a, signal)
#     return filtered


# def filter_signal(signal_all_leads):
#     X, Y = signal_all_leads.shape  # X filas, 12 columnas
#     leads__array = np.zeros((X, Y))  # Matriz destino con ceros

#     for col in range(Y):
#         signal = signal_all_leads[:, col]  # Extraer columna
#         # Verificar longitud mínima
#         min_length = int(400 * 600 / 1000)
#         if len(signal) < min_length:
#             print(f"Advertencia: Señal en lead {col} tiene longitud {len(signal)} < {min_length}. Saltando procesamiento.")
#             leads__array[:, col] = signal  # Dejar sin filtrar o manejar de otra forma
#             continue

#         signal = apply_median_filter(signal)
#         signal = bandpass_filter(signal)

#         leads__array[:, col] = signal

#     return leads__array

def preprocess_12_lead_signal(all_lead_signal):
    # all_lead_signal = filter_signal(all_lead_signal)

    all_lead_signal = Zero_pad_leads(all_lead_signal)
    all_lead_signal = paddedEcg_to_vcg(all_lead_signal)
    
    return all_lead_signal



# Agrupar edades en intervalos de 5 años
def age_group(age):
    return (age // 5) * 5

def obtain_balanced_train_dataset(path, negative_to_positive_ratio=1.0):
    """
    Selecciona registros positivos y negativos de una base de datos, con una proporción
    de negativos aproximadamente igual a negative_to_positive_ratio * len(positivos).
    
    Args:
        path (str): Ruta al directorio con los registros.
        negative_to_positive_ratio (float): Proporción deseada de negativos respecto a positivos (default=1.0).
    
    Returns:
        list: Lista de registros seleccionados (positivos + negativos).
    
    Raises:
        ValueError: Si no hay registros positivos en la base de datos.
    """
    # Obtener todos los registros
    records = find_records(path, '.hea')
    for i in range(len(records)):
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
            age_sex_distribution.append((age_group(age), sex))
    
    num_positives = len(positive_records)
    if num_positives == 0:
        raise ValueError("No hay registros positivos en la base de datos")
    
    # Distribución de positivos por combinación (edad en lustros, sexo)
    positive_distribution = Counter(age_sex_distribution)
    
    # Obtener candidatos negativos
    negative_candidates = [rec for rec in records if load_label(rec) == 0]
    
    # Agrupar negativos por combinación (edad en lustros, sexo)
    negative_by_combination = defaultdict(list)
    for rec in negative_candidates:
        head = load_header(rec)
        age = get_age(head)
        sex = get_sex(head)
        comb = (age_group(age), sex)
        # Solo consideramos combinaciones presentes en los positivos
        if comb in positive_distribution:
            negative_by_combination[comb].append(rec)
    
    # Barajar los negativos dentro de cada combinación para selección aleatoria
    for comb in negative_by_combination:
        random.shuffle(negative_by_combination[comb])
    
    # Calcular el número total deseado de negativos
    total_desired_negatives = math.ceil(negative_to_positive_ratio * num_positives)
    
    # Inicializar estructuras para la selección
    selected_negatives = []
    selected_counts = {comb: 0 for comb in positive_distribution}
    
    # Seleccionar negativos hasta alcanzar el total deseado o agotar candidatos
    while len(selected_negatives) < total_desired_negatives and any(negative_by_combination[comb] for comb in positive_distribution):
        # Encontrar la combinación con la menor proporción respecto a lo deseado
        min_ratio = float('inf')
        best_comb = None
        for comb in positive_distribution:
            if negative_by_combination[comb]:  # Si hay negativos disponibles
                desired = negative_to_positive_ratio * positive_distribution[comb]
                ratio_current = selected_counts[comb] / desired if desired > 0 else float('inf')
                if ratio_current < min_ratio:
                    min_ratio = ratio_current
                    best_comb = comb
        
        if best_comb is None:
            break  # No hay más combinaciones con negativos disponibles
        
        # Seleccionar un negativo de la combinación elegida
        neg_rec = negative_by_combination[best_comb].pop()
        selected_negatives.append(neg_rec)
        selected_counts[best_comb] += 1
    
    # Devolver la lista completa de registros seleccionados
    return positive_records + selected_negatives


import numpy as np
import os

# Data Augmentation Functions
def add_noise(ecg, noise_level=0.05):
    """Add Gaussian noise to ECG signal."""
    noise = np.random.normal(0, noise_level, ecg.shape)
    return ecg + noise

def time_shift(ecg, max_shift=200):
    """Shift ECG signal in time (circular shift)."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(ecg, shift, axis=-1)

def scale_amplitude(ecg, scale_range=(0.8, 1.2)):
    """Scale the amplitude of ECG signal."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return ecg * scale

def time_warp(ecg, warp_factor=0.1):
    """Apply time warping by stretching/compressing time axis."""
    n_samples, n_leads, signal_length = ecg.shape
    time_points = np.linspace(0, signal_length - 1, signal_length)
    warped_ecg = np.zeros_like(ecg)
    
    for i in range(n_samples):
        for j in range(n_leads):
            warp = np.random.uniform(-warp_factor, warp_factor)
            new_time = time_points * (1 + warp)
            new_time = np.clip(new_time, 0, signal_length - 1)
            warped_ecg[i, j] = np.interp(time_points, new_time, ecg[i, j])
    return warped_ecg

def augment_ecg_data(ecg_data, labels, n_augmentations=5):
    """
    Generate augmented ECG samples and corresponding labels.
    Args:
        ecg_data: Input ECG data of shape (n_samples, 12, 4096)
        labels: Corresponding labels of shape (n_samples,)
        n_augmentations: Number of augmented samples per original sample
    Returns:
        Augmented ECG data of shape (n_samples * (n_augmentations + 1), 12, 4096)
        Augmented labels of shape (n_samples * (n_augmentations + 1),)
    """
    n_samples, n_leads, signal_length = ecg_data.shape
    augmented_data = []
    augmented_labels = []
    
    for i in range(n_samples):
        original_ecg = ecg_data[i:i+1]  # Shape (1, 12, 4096)
        original_label = labels[i]
        augmented_data.append(original_ecg)  # Include original sample
        augmented_labels.append(original_label)
        
        for _ in range(n_augmentations):
            aug_ecg = original_ecg.copy()
            
            # Randomly apply augmentations
            if np.random.rand() > 0.3:
                aug_ecg = add_noise(aug_ecg, noise_level=0.05)
            if np.random.rand() > 0.3:
                aug_ecg = time_shift(aug_ecg, max_shift=200)
            if np.random.rand() > 0.3:
                aug_ecg = scale_amplitude(aug_ecg, scale_range=(0.8, 1.2))
            if np.random.rand() > 0.3:
                aug_ecg = time_warp(aug_ecg, warp_factor=0.1)
            
            augmented_data.append(aug_ecg)
            augmented_labels.append(original_label)  # Same label for augmented sample
    
    augmented_data = np.vstack(augmented_data)
    augmented_labels = np.array(augmented_labels)
    return augmented_data, augmented_labels


# Filterning Functions
from scipy.signal import medfilt
def remove_baseline_wander(signal, fs, window_time=0.2):
    window_size = int(fs * window_time)
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    # Apply median filter
    y = medfilt(signal, kernel_size=window_size)
    # Subtract baseline wander
    filt_signal = signal - y
    return filt_signal

import pywt # Wavelet package

def wavelet_filter(senal_entrada, wavelet='coif4', nivel=2):

    # Sacada de un libro de procesamiento de electrocardiogramas

    """
    Filtra una señal utilizando transformada Wavelet.

    Parámetros:
        senal_entrada (array-like): Señal de entrada a filtrar.
        wavelet (str): Familia de wavelet a usar (por defecto 'coif4').
        nivel (int): Número de niveles de descomposición (por defecto 7).
    
    Retorna:
        senal_filtrada (numpy array): Señal filtrada reconstruida.
    """
    # Descomponer la señal usando wavelet y niveles especificados
    coeficientes = pywt.wavedec(senal_entrada, wavelet, level=nivel)
    
    # Filtrado: Eliminamos los detalles (coeficientes 'd') manteniendo aproximaciones ('a')
    # Nota: Esto corresponde al componente de baja frecuencia.
    for i in range(1, len(coeficientes)):
        coeficientes[i] = np.zeros_like(coeficientes[i])
    
    # Reconstrucción de la señal
    senal_filtrada = pywt.waverec(coeficientes, wavelet)
    
    # Ajustar la longitud en caso de que haya cambiado
    senal_filtrada = senal_filtrada[:len(senal_entrada)]
    
    
    return senal_filtrada

def filter_median_wavelet(ecg_signal, fs=500, level=2, wavelet='coif4'):
    # First remove baseline wander and then filter
    ecg_signal = remove_baseline_wander(signal = ecg_signal, fs=fs, window_time=0.2)
    ecg_signal = wavelet_filter(ecg_signal, wavelet = wavelet, nivel=level)
    return ecg_signal



# VCG to ECG

def ecg_to_vcg(ecg, tr='dower'):
    # Dimensiones ECG (input):   (5000, 12)
    # Dimensiones VCG (output):   (1000, 3)
    if tr == 'dower':
        T = np.array([[-0.172, -0.074,  0.122,  0.231, 0.239, 0.194,  0.156, -0.010],
                    [0.057,  -0.019, -0.106, -0.022, 0.041, 0.048, -0.227,  0.887],
                    [-0.229,  -0.310, -0.246, -0.063, 0.055, 0.108,  0.022,  0.102]])
    if tr == 'kors':
        T = np.array([[ -0.13, 0.05, -0.01, 0.14, 0.06, 0.54, 0.38, -0.07],
                      [0.06, -0.02, -0.05, 0.06, -0.17, 0.13, -0.07, 0.93],
                      [-0.43, -0.06, -0.14, -0.20, -0.11, 0.31, 0.11, -0.23]])
    # Seleccionar las columnas apropiadas para ecg_1 y ecg_2
    #ecg = np.transpose(ecg, (0,2,1))
    ecg_1 = ecg[:, 6:]
    ecg_2 = ecg[:, :2]

   

    # Concatenar ecg_1 y ecg_2 a lo largo del eje 2 (columnas)
    ecg_red = np.concatenate([ecg_1, ecg_2], axis=1)

    ecg_red = np.transpose(ecg_red,(1,0) )


    # Realizar la multiplicación matricial
    vcg = np.matmul(T, ecg_red)

    vcg = np.transpose(vcg, (1, 0))  # Transpose to match expected output shape (3, 4096)

    return vcg



# Padded ECG signal to VCG (use this directly)
def paddedEcg_to_vcg(ecg_padded):
    # Input: ECG signal (4096, 12)
    # Output: VCG signal (4096, 3)

    # # Initialize filtered signal container
    # filtered = np.zeros_like(ecg_padded)
    # # Filter
    # for lead_idx in range(ecg_padded.shape[1]):
    #     filtered[:, lead_idx] = filter_median_wavelet(ecg_signal = ecg_padded[:, lead_idx], fs = 400)
    
    # VCG transform
    vcg = ecg_to_vcg(ecg_padded)


    return vcg
