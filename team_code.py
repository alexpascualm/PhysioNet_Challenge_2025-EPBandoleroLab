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


import numpy as np
import pandas as pd


from collections import Counter, defaultdict
import math
import random
from scipy.signal import resample_poly


PROB_THRESHOLD = 0.6
BATCH_SIZE = 128

################################################################################
#
# TODO
#
################################################################################

def set_all_seeds(seed=42):
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # Deterministic operations in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For reproducibility in DataLoader
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker

# Llama a la función al principio
seed = 123
worker_seed_fn = set_all_seeds(seed)



################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
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


    train_data_records = []  # Lista para almacenar los registros de datos

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        
        # record = os.path.join(data_folder, records[i]) # Not needed if obtain_balanced_train_dataset() used
        record = records[i]

        label = load_label(record)
        signal_data = load_signals(record)
        signal = signal_data[0]
        sampling_frequency = get_sampling_frequency(load_header(record))
        source = load_source(record)

        if sampling_frequency != 400:
            signal = resample_poly(signal, 400, sampling_frequency, axis=0)

        processed_signals = preprocess_12_lead_signal(signal) 
        
        # Añadir una entrada por cada señal procesada
        for j, processed_signal in enumerate(processed_signals):
                train_data_records.append({
                    'record': record,
                    'variant_index': j,  # índice dentro del listado devuelto por preprocess
                    'signal': processed_signal,
                    'label': label,
                    'probability': get_probability(load_header(record), allow_missing=True),
                    'source': source,
                    'age': load_age(record),
                    'sex': load_sex(record),
                    'sampling_frequency': sampling_frequency
                })
                
    
    # Convertir la lista de diccionarios en un DataFrame
    train_df = pd.DataFrame(train_data_records)
    
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    train_and_save_model(train_df,model_folder,obtain_test_metrics=False)
    
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
    signal_segments = preprocess_12_lead_signal(signal)

    segments_tensor = np.stack(signal_segments)
    labels = np.repeat(label, len(signal_segments))

    test_dataset = ECGDataset(segments_tensor, labels)
    test_loader = DataLoader(test_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    segment_probabilities = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            prob = torch.sigmoid(outputs)
            segment_probabilities.append(prob.item())

    probability_output = np.mean(segment_probabilities)
    binary_output = probability_output > PROB_THRESHOLD

    return binary_output, probability_output


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

################################################################################
#
# TODO Arquitecura del modelo y funciones de entrenamiento
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
        # ecg = (ecg - ecg.mean(axis=0)) / (ecg.std(axis=0) + 1e-8)
        # ecg = (ecg - ecg.mean(axis=0)) 

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


# Crear subconjuntos por record
def get_data_by_patients(df, patient_list):
    df_subset = df[df['record'].isin(patient_list)]
    X = np.stack(df_subset['signal'].tolist(), axis=0)
    y = np.array(df_subset['label'].tolist(), dtype=bool)
    return X, y

# 3. Función de entrenamiento y guardado
def train_and_save_model(df, model_folder,obtain_test_metrics):
    # Configuración inicial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("GPU Available")
    else:
        print("GPU not available")

    
    # Agrupar por record
    record = df['record'].unique()
    train_patients, valtest_patients = train_test_split(
    record, test_size=0.3, random_state=42
    )

    if obtain_test_metrics:
        val_patients, test_patients = train_test_split(
            valtest_patients, test_size=0.5, random_state=42
        )

        X_test, y_test = get_data_by_patients(df, test_patients)
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, worker_init_fn=worker_seed_fn
        )
    else:
        val_patients = valtest_patients
        test_patients = []

    
    X_train, y_train = get_data_by_patients(df, train_patients)
    X_val, y_val = get_data_by_patients(df, val_patients)
    
    # Crear datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
        
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True,num_workers=4, worker_init_fn=worker_seed_fn)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE,shuffle=False,num_workers=4, worker_init_fn=worker_seed_fn)
    
    
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
            print("New best model found, saving...")
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
        if train_challenge_score - val_challenge_score > 0.1 or train_auprc - val_auprc > 0.06:  # Umbral arbitrario para detectar overfitting
            print("Warning: Potential overfitting detected (Train Challenge metrics significantly higher than Val)")
            break

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

################################################################################
#
# TODO
#
################################################################################

def Zero_pad_leads(arr, target_length=1024):
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

def adjust_length_ecg_2048(arr):
    
    target_length=2048
    X, Y = arr.shape  # X filas, 12 columnas
    signal1 = np.zeros((target_length, Y))  # Matriz destino con ceros
    signal2 = np.zeros((target_length, Y))  # Matriz destino con ceros
    signals = list()
    
    for col in range(Y):
        col_data = arr[:, col]  # Extraer columna
        length = len(col_data)
        
        if length < target_length:
            break

        elif length > 1.5*target_length: # Its bigger than target but not double, overlapping
            signal1[:, col] = col_data[:target_length] # Take the first target_length
            signal2[:, col] = col_data[-target_length:] # Take the last
        
        else:
            signal1[:, col] = col_data[:target_length] # Only take the first
            signal2 = 0
            
    signals.append(signal1)
    if isinstance(signal2,np.ndarray):
        signals.append(signal2)
    return signals

def adjust_length_ecg_1024(arr):
    
    target_length = 1024
    X, Y = arr.shape  # X filas, 12 columnas
    signal1 = np.zeros((target_length, Y))  # Matriz destino con ceros
    signal2 = np.zeros((target_length, Y))  # Matriz destino con ceros
    signal3 = np.zeros((target_length, Y))  # Matriz destino con ceros
    signal4 = np.zeros((target_length, Y))  # Matriz destino con ceros
    signals = list()
    
    for col in range(Y):
        col_data = arr[:, col]  # Extraer columna
        length = len(col_data)
        
        if length < target_length:
            print("Señal corta! Registra más")
            break

        elif length >= 4*target_length: # If its bigger than two times we do two splits
            signal1[:, col] = col_data[:target_length]
            signal2[:, col] = col_data[target_length:2*target_length]
            signal3[:, col] = col_data[2*target_length:3*target_length]
            signal4[:, col] = col_data[3*target_length:4*target_length]

        elif length >= 3*target_length: # If its bigger than two times we do two splits
            signal1[:, col] = col_data[:target_length]
            signal2[:, col] = col_data[target_length:2*target_length]
            signal3[:, col] = col_data[2*target_length:3*target_length]
            signal4 = 0

        elif length > 1.5*target_length: # Its bigger than target but not double, overlapping
            signal1[:, col] = col_data[:target_length] # Take the first target_length
            signal2[:, col] = col_data[-target_length:] # Take the last
            signal3 = 0
            signal4 = 0
        
        else:
            signal1[:, col] = col_data[:target_length] # Only take the first
            signal2 = 0
            signal3 = 0
            signal4 = 0

    signals.append(signal1) 
    
    if isinstance(signal4,np.ndarray):
        signals.extend([signal2, signal3, signal4])
    elif isinstance(signal3,np.ndarray):
        signals.extend([signal2, signal3])
    elif isinstance(signal2,np.ndarray):
        signals.append(signal2)

    return signals


# Filtering function
import pywt # Wavelet package
def wavelet_ecg_filter(signal, wavelet='db4', mode='symmetric', 
                   remove_approx=True, remove_details=[8,7]):

    # Use in chagas code -> filtered_ecg_signal = waveltet_filter(ecg_signal)
    # INPUT: Lead signal (2048, 1)
    # OUTPUT: Filtered lead signal (2048,1)
    # Removes baseline wander and high frequency noise preserving the morphology 
    #   using wavelet decomposition

    """
    Parameters:
    -----------
    signal : array-like
        Input signal to process
    wavelet : str, optional
        Wavelet to use (default: 'db4')
    mode : str, optional
        Signal extension mode (default: 'symmetric')
    remove_approx : bool, optional
        Whether to remove the approximation coefficients (default: True)
    remove_details : list or None, optional
        Which detail levels to remove (e.g., [1,2] removes D1 and D2) (default: [8,7])
    plot_Flag: bool, optional
        Whether to plot the components (default: False)
    figsize : tuple, optional
        Figure size (default: (30, 5))
        
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet=wavelet, mode=mode, level=8)
    levels = len(coeffs) - 1
    
    # Initialize dictionary to store components
    components = {}
    
    # Create modified coefficients for filtering
    filtered_coeffs = [c.copy() for c in coeffs]
    
    # Remove specified components
    if remove_approx:
        filtered_coeffs[0] = np.zeros_like(filtered_coeffs[0])
    
    if remove_details is not None:
        for level in remove_details:
            if 1 <= level <= levels:
                filtered_coeffs[level] = np.zeros_like(filtered_coeffs[level])
    
    # Reconstruct filtered signal
    filtered_signal = pywt.waverec(filtered_coeffs, wavelet=wavelet, mode=mode)
    filtered_signal = filtered_signal[:len(signal)]  # Match original length
    
    return filtered_signal



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



def signal_segment_to_model_input(padded_ecg, vcg=True, filter=False, normalization = 'normalize'):
    if filter:
        # Initialize filtered signal container
        filtered = np.zeros_like(padded_ecg)
        # Filter
        for lead_idx in range(padded_ecg.shape[1]):
            filtered[:, lead_idx] = wavelet_ecg_filter(signal = padded_ecg[:, lead_idx])
    else:
        filtered = padded_ecg
    
    # Normalize or center before VCG, to preserve spatial relations in VCG
    if normalization == 'normalize':
        filtered = (filtered - filtered.mean(axis=0)) / (filtered.std(axis=0) + 1e-8)
    if normalization == 'center':
        filtered = (filtered - filtered.mean(axis=0))
    
    # Transform to VCG
    if vcg:
        vcg = ecg_to_vcg(filtered)

    return vcg



def preprocess_12_lead_signal(all_lead_signal):
    # all_lead_signal: np.ndarray con shape [12, N]

    # Paso 1: Cortar o segmentar en múltiples señales de shape [12, N]
    signal_segments = adjust_length_ecg_1024(all_lead_signal) #  Lista de arrays [12, N]

    # Paso 2: Convertir cada segmento a VCG (u otra representación)
    processed_segments = [
       signal_segment_to_model_input(segment, vcg = True, filter=False, normalization='normalize')
        for segment in signal_segments
    ]
    

    return processed_segments  # Lista de arrays transformados

################################################################################
#
# TODO Comment Data Selection Functions 
#
################################################################################

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
        if load_label(rec) == 1: # Seleccionar solo samitrops
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
    for rec in negative_candidates: #Seleccionar solo europeos o tambien code (Si son PTB resamplear)?
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






