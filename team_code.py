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
VCG_TRANSFORM = False
INPUT_CHANNELS = 3 if VCG_TRANSFORM else 12
SEGMENTS_LENGTH = 1024  # Cambia a 2048 si quieres usar segmentos de 2048 muestras

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
        
        record = records[i]

        label = load_label(record)
        signal_data = load_signals(record)
        signal = signal_data[0]
        sampling_frequency = get_sampling_frequency(load_header(record))
        source = load_source(record)

        processed_signals = preprocess_12_lead_signal(signal,sampling_frequency, source) 
        
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

    source = load_source(record)
    sampling_frequency = get_sampling_frequency(load_header(record))

    signal_segments = preprocess_12_lead_signal(signal,sampling_frequency, source)
                                    
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


# def run_model(record, model, verbose):
#     signal_data = load_signals(record)
#     signal = signal_data[0]
    
#     source = load_source(record)
#     sampling_frequency = get_sampling_frequency(load_header(record))

#     signal = standardize_ecg_signal(signal, sampling_frequency,source)
         
#     TMD = calculate_tmd_from_raw_ecg(signal.T, 400)


#     # # Nombres reales de las 12 derivaciones estándar
#     # lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
#     #             "V1", "V2", "V3", "V4", "V5", "V6"]
#     # # Crear DataFrame
#     # df = pd.DataFrame(signal, columns=lead_names)
#     # # Formatear el nombre del archivo con TMD
#     # tmd_str = f"{TMD:.2f}"
#     # record = record.split('/')[-1] 
#     # filename = f"{source}_{record}_TMD_{tmd_str}.csv"
#     # print(filename)
#     # output_dir = "/home/jamon/alejandropm/PhysioNet_Challenge/PhysioNet_Challenge_2025-EPBandoleroLab/Prueba"
#     # output_path = os.path.join(output_dir, filename)
#     # print(output_path)
#     # # Guardar
#     # df.to_csv(output_path, index=False)


#     binary_output = 1 if TMD > 0.5 else 0
#     probability_output = TMD

#     return binary_output, probability_output


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

################################################################################
#
# TODO Arquitecura del modelo
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

        return torch.tensor(ecg.T, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Cambio clave: solo añade una dimensión de batch al principio.
        # pe tiene forma [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x tiene forma [batch_size, seq_len, d_model]
        # self.pe tiene forma [1, max_len, d_model]
        # El slicing selecciona las posiciones correctas y broadcasting se encarga del resto.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ChagasClassifier(nn.Module):
    def __init__(self, input_len=1024): # Asumimos una longitud de entrada, ej. 1024
        super().__init__()
        
        # 1. Extractor de características CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=18, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # len -> len/2
            nn.Conv1d(32, 64, kernel_size=18, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # len -> len/4
            nn.Conv1d(64, 128, kernel_size=18, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4), # len -> len/16
            nn.Conv1d(128, 256, kernel_size=18, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4), # len -> len/64
        )
        
        # Con una entrada de 1024, la longitud de la secuencia de salida de la CNN sería:
        # 1024 / (2*2*4*4) = 1024 / 64 ≈ 16 -> carac_len
        # El número de características (canales) es 256.
        # Así que la salida de la CNN es [batch, 256, carac_len]

        # 2. Componentes del Transformer
        d_model = 256  # El tamaño de la característica debe coincidir con los canales de la CNN
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Token [CLS]

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=513) # max_len > carac_len
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=16, # 8 cabezales de atención
            dim_feedforward=1024, # Capa feedforward más pequeña
            dropout=0.2,
            batch_first=True  # ¡Muy importante para manejar las dimensiones fácilmente!
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), # d_model es 256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 1. Pasar por la CNN para obtener una secuencia de características
        cnn_features = self.cnn(x)  # Shape: [batch, 256, seq_len_out], ej: [batch, 256, 16]

        transformer_in = cnn_features.permute(0, 2, 1)
        
        # Añadir el token [CLS] al inicio de cada secuencia en el batch
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_in = torch.cat([cls_tokens, transformer_in], dim=1) # Nueva secuencia tiene longitud 17

        # El positional encoding también debe poder manejar una secuencia más larga
        transformer_in = self.pos_encoder(transformer_in) # Asegúrate que max_len en PositionalEncoding es > 17

        attn_output = self.transformer(transformer_in)

        # 5. Selecciona SOLO la salida del token [CLS] (que está en la posición 0)
        cls_output = attn_output[:, 0] # Shape: [batch, 256]
        # 6. Clasificar
        return self.classifier(cls_output)

    
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

        # Cálculo del factor alpha dinámico
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Aplicar Focal Loss
        # F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

    
################################################################################
#
# TODO Entrenamiento del modelo
#
################################################################################

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
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / pos_count

    pos_weight = torch.tensor([pos_weight], device=device)  
    # criterion = nn.BCEWithLogitsLoss()
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
        
        # # Ajustar el learning rate basado en el challenge_score de validación
        # scheduler.step(val_challenge_score)
        
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
            # signal1 = Zero_pad_leads(arr,target_length=2048)
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
        #    signal1 = Zero_pad_leads(arr,target_length=1024)
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
# import pywt # Wavelet package
# def wavelet_ecg_filter(signal, wavelet='db4', mode='symmetric', 
#                    remove_approx=True, remove_details=[8,7]):

#     # Use in chagas code -> filtered_ecg_signal = waveltet_filter(ecg_signal)
#     # INPUT: Lead signal (2048, 1)
#     # OUTPUT: Filtered lead signal (2048,1)
#     # Removes baseline wander and high frequency noise preserving the morphology 
#     #   using wavelet decomposition

#     """
#     Parameters:
#     -----------
#     signal : array-like
#         Input signal to process
#     wavelet : str, optional
#         Wavelet to use (default: 'db4')
#     mode : str, optional
#         Signal extension mode (default: 'symmetric')
#     remove_approx : bool, optional
#         Whether to remove the approximation coefficients (default: True)
#     remove_details : list or None, optional
#         Which detail levels to remove (e.g., [1,2] removes D1 and D2) (default: [8,7])
#     plot_Flag: bool, optional
#         Whether to plot the components (default: False)
#     figsize : tuple, optional
#         Figure size (default: (30, 5))
        
#     """
#     # Perform wavelet decomposition
#     coeffs = pywt.wavedec(signal, wavelet=wavelet, mode=mode, level=8)
#     levels = len(coeffs) - 1
    
#     # Initialize dictionary to store components
#     components = {}
    
#     # Create modified coefficients for filtering
#     filtered_coeffs = [c.copy() for c in coeffs]
    
#     # Remove specified components
#     if remove_approx:
#         filtered_coeffs[0] = np.zeros_like(filtered_coeffs[0])
    
#     if remove_details is not None:
#         for level in remove_details:
#             if 1 <= level <= levels:
#                 filtered_coeffs[level] = np.zeros_like(filtered_coeffs[level])
    
#     # Reconstruct filtered signal
#     filtered_signal = pywt.waverec(filtered_coeffs, wavelet=wavelet, mode=mode)
#     filtered_signal = filtered_signal[:len(signal)]  # Match original length
    
#     return filtered_signal



# VCG to ECG
def ecg_to_vcg(ecg, tr='dower'):
    """
    Convierte una señal de ECG de 12 derivaciones en una representación VCG de 3 canales 
    usando la transformación de Dower o Kors.

    Parameters
    ----------
    ecg : np.ndarray
        Array de forma (N, 12) donde N es el número de muestras y 12 las derivaciones del ECG.
    tr : str, optional
        Tipo de transformación a utilizar: 'dower' (por defecto) o 'kors'.

    Returns
    -------
    vcg : np.ndarray
        Array transformado de forma (N, 3) correspondiente a las componentes X, Y, Z del VCG.
    """
    if tr == 'dower':
        T = np.array([[-0.172, -0.074,  0.122,  0.231, 0.239, 0.194,  0.156, -0.010],
                      [ 0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227,  0.887],
                      [-0.229, -0.310, -0.246, -0.063, 0.055, 0.108,  0.022,  0.102]])
    elif tr == 'kors':
        T = np.array([[-0.13, 0.05, -0.01, 0.14, 0.06, 0.54, 0.38, -0.07],
                      [ 0.06, -0.02, -0.05, 0.06, -0.17, 0.13, -0.07,  0.93],
                      [-0.43, -0.06, -0.14, -0.20, -0.11, 0.31,  0.11, -0.23]])
    
    ecg_1 = ecg[:, 6:]  # Derivaciones V1-V6
    ecg_2 = ecg[:, :2]  # Derivaciones I y II
    ecg_red = np.concatenate([ecg_1, ecg_2], axis=1)  # Reordenar derivaciones

    ecg_red = ecg_red.T  # Transposición para multiplicación matricial
    vcg = np.matmul(T, ecg_red).T  # Transformación y transposición final

    return vcg


def standardize_ecg_signal(signal: np.ndarray, sampling_frequency: int, source: str) -> np.ndarray:
    """
    Estandariza una señal de ECG a 12 derivaciones, ajustando el orden de las derivaciones
    y la frecuencia de muestreo a 400 Hz.

    Parameters
    ----------
    signal : np.ndarray
        Señal ECG con forma (N, 12), donde N es el número de muestras.
    sampling_frequency : int
        Frecuencia de muestreo original de la señal.
    source : str
        Nombre de la base de datos fuente (ej. 'PTB-XL', 'SamiTrop', 'Code-15').

    Returns
    -------
    np.ndarray
        Señal ECG con derivaciones ordenadas y muestreada a 400 Hz.
    """
    if source == "PTB-XL":
        # Intercambiar aVR y aVL para que coincidan con el orden estándar
        signal[3, :], signal[4, :] = signal[4, :].copy(), signal[3, :].copy()

    if sampling_frequency != 400:
        signal = resample_poly(signal, 400, sampling_frequency, axis=0)

    return signal


def signal_segment_to_model_input(padded_ecg, vcg=True, filter=False, normalization='normalize'):
    """
    Procesa un segmento de señal ECG aplicando filtrado, normalización y transformación opcional a VCG.

    Parameters
    ----------
    padded_ecg : np.ndarray
        Segmento ECG de forma (N, 12).
    vcg : bool, optional
        Si True, convierte la señal a representación VCG. Por defecto es True.
    filter : bool, optional
        Si True, aplica un filtro. Por defecto es False.
    normalization : str, optional
        Método de normalización: 'normalize' (media 0, varianza 1) o 'center' (solo media 0).

    Returns
    -------
    np.ndarray
        Segmento procesado, ya sea como ECG (N, 12) o VCG (N, 3).
    """
    if filter:
        # Placeholder para un filtro real
        filtered = np.zeros_like(padded_ecg)
        for lead_idx in range(padded_ecg.shape[1]):
            filtered[:, lead_idx] = filtered[:, lead_idx]  # Sustituir por filtro real
    else:
        filtered = padded_ecg

    if normalization == 'normalize':
        filtered = (filtered - filtered.mean(axis=0)) / (filtered.std(axis=0) + 1e-8)
    elif normalization == 'center':
        filtered = filtered - filtered.mean(axis=0)

    if vcg:
        filtered = ecg_to_vcg(filtered)

    return filtered


def preprocess_12_lead_signal(all_lead_signal, sampling_frequency, source):
    """
    Preprocesa una señal ECG de 12 derivaciones: estandariza frecuencia y orden, 
    la segmenta y la convierte a VCG si se desea.

    Parameters
    ----------
    all_lead_signal : np.ndarray
        Señal ECG completa de forma (N, 12), donde N es el número de muestras.
    sampling_frequency : int
        Frecuencia de muestreo original de la señal.
    source : str
        Nombre de la base de datos fuente (ej. 'PTB-XL', 'SamiTrop', 'Code-15').

    Returns
    -------
    list of np.ndarray
        Lista de segmentos (N,12) procesados, transformados según configuración.
    """
    # Paso 0: Estandarizar señal
    all_lead_signal = standardize_ecg_signal(all_lead_signal, sampling_frequency, source)

    # Paso 1: Segmentar según longitud deseada
    if SEGMENTS_LENGTH == 2048:
        signal_segments = adjust_length_ecg_2048(all_lead_signal)
    elif SEGMENTS_LENGTH == 1024:
        signal_segments = adjust_length_ecg_1024(all_lead_signal)


    # Paso 2: Procesar cada segmento (VCG, normalización, etc.)
    processed_segments = [
        signal_segment_to_model_input(segment, vcg=VCG_TRANSFORM, filter=False, normalization='normalize')
        for segment in signal_segments
    ]
    return processed_segments

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

################################################################################
#
# TODO TMD
#
################################################################################


# import numpy as np
# from scipy.signal import butter, lfilter, find_peaks
# from scipy.ndimage import median_filter

# def _filter_ecg(data, sampling_rate, low_cutoff=0.5, high_cutoff=40.0, order=4):
#     """
#     Aplica filtros paso bajo y paso alto al ECG.
#     Equivalente a la lógica de filtrado en BRAVEHEART.
#     """
#     nyquist = 0.5 * sampling_rate
    
#     # Filtro paso alto (para deriva de línea de base)
#     b_high, a_high = butter(order, low_cutoff / nyquist, btype='high')
#     filtered_data = lfilter(b_high, a_high, data, axis=1)
    
#     # Filtro paso bajo (para ruido de alta frecuencia)
#     b_low, a_low = butter(order, high_cutoff / nyquist, btype='low')
#     filtered_data = lfilter(b_low, a_low, filtered_data, axis=1)
    
#     return filtered_data

# def _ecg_to_vcg(ecg_leads):
#     """
#     Transforma 8 derivaciones de ECG a VCG usando la matriz de Kors.
#     """
#     # Matriz de Kors
#     K = np.array([
#         [-0.13,  0.05, -0.01,  0.14,  0.06,  0.54,  0.38, -0.07],
#         [ 0.06, -0.02, -0.05,  0.06, -0.17,  0.13, -0.07,  0.93],
#         [-0.43, -0.06, -0.14, -0.20, -0.11,  0.31,  0.11, -0.23]
#     ])
    
#     # Las derivaciones deben estar en el orden: I, II, V1, V2, V3, V4, V5, V6
#     E = np.vstack([
#         ecg_leads['I'], ecg_leads['II'], ecg_leads['V1'], ecg_leads['V2'],
#         ecg_leads['V3'], ecg_leads['V4'], ecg_leads['V5'], ecg_leads['V6']
#     ])
    
#     vcg = K @ E
#     X, Y, Z = vcg[0, :], vcg[1, :], vcg[2, :]
#     VM = np.sqrt(X**2 + Y**2 + Z**2)
    
#     return {'X': X, 'Y': Y, 'Z': Z, 'VM': VM}

# def _find_r_peaks(vm_signal, sampling_rate):
#     """
#     Encuentra los picos R en la señal de magnitud del vector.
#     """
#     # Umbral al percentil 95, como en BRAVEHEART
#     height_threshold = np.percentile(vm_signal, 95)
#     # Distancia mínima basada en una FCM máxima de 180 lpm
#     min_peak_distance = (60 / 180) * sampling_rate
    
#     peaks, _ = find_peaks(vm_signal, height=height_threshold, distance=min_peak_distance)
#     return peaks

# def _annotate_single_beat(vm_signal, r_peak, sampling_rate):
#     """
#     Anota un único latido (Q, S, T, Tend) a partir de su señal de VM y pico R.
#     Esta es una versión simplificada de la lógica de annoMF.m.
#     """
#     # Estimar ancho de QRS usando un filtro de mediana
#     mf_length = int(0.04 * sampling_rate) # filtro de 40ms
#     if mf_length % 2 == 0: mf_length += 1
    
#     smoothed_vm = median_filter(vm_signal, size=mf_length)
#     peak_val = smoothed_vm[r_peak]
#     width_threshold = 0.20 * peak_val

#     # Búsqueda de inicio y fin del QRS
#     qrs_search_win_ms = 100 # 100ms antes y después del R
#     search_radius = int(qrs_search_win_ms / 1000 * sampling_rate)
    
#     start_search = max(0, r_peak - search_radius)
#     end_search = min(len(smoothed_vm), r_peak + search_radius)

#     crossings_start = np.where(smoothed_vm[start_search:r_peak] < width_threshold)[0]
#     q_point = start_search + crossings_start[-1] + 1 if len(crossings_start) > 0 else start_search
    
#     crossings_end = np.where(smoothed_vm[r_peak:end_search] < width_threshold)[0]
#     s_point = r_peak + crossings_end[0] if len(crossings_end) > 0 else end_search

#     # Búsqueda de la onda T
#     st_start_ms = 100 # 100ms después del fin del QRS
#     t_win_start = s_point + int(st_start_ms / 1000 * sampling_rate)
    
#     # La ventana de búsqueda de T finaliza 45% del intervalo RR estimado (asumiendo 60 lpm por defecto)
#     rr_interval_est = int(sampling_rate) 
#     t_win_end = min(len(vm_signal), t_win_start + int(0.45 * rr_interval_est))

#     if t_win_start >= t_win_end:
#         return q_point, s_point, s_point, t_win_end

#     # Encontrar el pico de la onda T
#     t_segment = vm_signal[t_win_start:t_win_end]
#     if len(t_segment) == 0:
#         return q_point, s_point, s_point, t_win_end
        
#     t_peak_relative = np.argmax(t_segment)
#     t_peak = t_win_start + t_peak_relative

#     # Encontrar fin de T con el método de energía
#     vm_derivative = np.gradient(vm_signal)
#     energy_segment = vm_signal[t_peak:t_win_end]
#     energy_derivative_segment = vm_derivative[t_peak:t_win_end]
    
#     energy_signal = np.zeros_like(energy_segment)
#     for i in range(1, len(energy_signal)):
#         if energy_derivative_segment[i] < 0:
#             energy_signal[i] = energy_signal[i-1] - energy_segment[i]
#         else:
#             energy_signal[i] = energy_signal[i-1] + energy_segment[i]
            
#     tend_relative = np.argmin(energy_signal) if len(energy_signal) > 0 else len(energy_segment) - 1
#     tend_point = t_peak + tend_relative
    
#     return q_point, s_point, t_peak, tend_point

# def _create_median_beat(ecg_data, r_peaks, sampling_rate):
#     """
#     Crea un latido mediano alineando todos los latidos por sus picos R.
#     """
#     window_before = int(0.2 * sampling_rate) # 200ms
#     window_after = int(0.6 * sampling_rate)  # 600ms
#     beat_length = window_before + window_after
    
#     num_beats = len(r_peaks)
#     num_leads = ecg_data.shape[0]
    
#     beat_stack = np.zeros((num_beats, num_leads, beat_length))
#     beat_stack[:] = np.nan
    
#     valid_beats = 0
#     for i, r_peak in enumerate(r_peaks):
#         start = r_peak - window_before
#         end = r_peak + window_after
        
#         if start >= 0 and end < ecg_data.shape[1]:
#             beat_stack[i, :, :] = ecg_data[:, start:end]
#             valid_beats += 1
            
#     if valid_beats < 3: # Se necesitan al menos unos pocos latidos para un mediano fiable
#         print("Advertencia: No hay suficientes latidos válidos para crear un latido mediano.")
#         return None
        
#     median_beat_data = np.nanmedian(beat_stack, axis=0)
    
#     # El nuevo "pico R" está en el centro de la ventana
#     median_r_peak = window_before
    
#     return median_beat_data, median_r_peak

# def calculate_tmd(ecg_leads, s_point, tend_point):
#     """
#     Calcula la Dispersión de la Morfología de la Onda T (TMD) a partir de 8 derivaciones del ECG.
#     (Función de la respuesta anterior, ligeramente adaptada para este contexto)
#     """
#     required_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#     if not all(lead in ecg_leads for lead in required_leads):
#         raise ValueError(f"El diccionario 'ecg_leads' debe contener las siguientes claves: {required_leads}")

#     if not (0 <= s_point < tend_point < len(ecg_leads['I'])):
#         print("Advertencia: Los puntos fiduciales del latido mediano (s_point, tend_point) son inválidos.")
#         return np.nan
        
#     X = np.array([
#         ecg_leads['I'][s_point : tend_point + 1],
#         ecg_leads['II'][s_point : tend_point + 1],
#         ecg_leads['V1'][s_point : tend_point + 1],
#         ecg_leads['V2'][s_point : tend_point + 1],
#         ecg_leads['V3'][s_point : tend_point + 1],
#         ecg_leads['V4'][s_point : tend_point + 1],
#         ecg_leads['V5'][s_point : tend_point + 1],
#         ecg_leads['V6'][s_point : tend_point + 1]
#     ])

#     if X.shape[1] < 2: # Se necesita al menos 2 puntos para SVD
#         print("Advertencia: El segmento de la onda T es demasiado corto para el análisis.")
#         return np.nan

#     try:
#         U, s, Vt = np.linalg.svd(X, full_matrices=False)
#     except np.linalg.LinAlgError:
#         print("Error: La descomposición SVD falló.")
#         return np.nan
        
#     Ut = U[:, :2]
#     S_diag = np.diag(s[:2])
#     W = (Ut @ S_diag).T
#     W = np.delete(W, 2, axis=1)

#     num_leads_remaining = W.shape[1]
#     angle_matrix = np.zeros((num_leads_remaining, num_leads_remaining))

#     for i in range(num_leads_remaining):
#         for j in range(i + 1, num_leads_remaining):
#             wi = W[:, i]
#             wj = W[:, j]
            
#             dot_product = np.dot(wi, wj)
#             norm_product = np.linalg.norm(wi) * np.linalg.norm(wj)
            
#             # Evitar división por cero si una derivación es plana
#             if norm_product == 0:
#                 angle_rad = np.pi / 2 # 90 grados si son ortogonales
#             else:
#                 # Acos es más estable para este cálculo de ángulo
#                 cos_angle = dot_product / norm_product
#                 angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
#             angle_matrix[i, j] = np.degrees(angle_rad)

#     upper_triangle_angles = angle_matrix[np.triu_indices(num_leads_remaining, k=1)]
    
#     return np.mean(upper_triangle_angles) if len(upper_triangle_angles) > 0 else 0.0

# def calculate_tmd_from_raw_ecg(ecg_data, sampling_rate):
#     """
#     Función principal que calcula el TMD a partir de datos brutos de un ECG de 12 derivaciones.

#     Parámetros:
#     ----------
#     ecg_data : np.array
#         Un array de NumPy de forma (12, N) con los datos del ECG.
#         El orden de las derivaciones debe ser: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
#     sampling_rate : int
#         La frecuencia de muestreo del ECG en Hz.

#     Devuelve:
#     -------
#     float
#         El valor del TMD calculado en grados, o np.nan si el cálculo no fue posible.
#     """
#     if ecg_data.shape[0] != 12:
#         raise ValueError("La entrada 'ecg_data' debe tener 12 filas (derivaciones).")

#     # 1. Filtrar el ECG
#     filtered_ecg_data = _filter_ecg(ecg_data, sampling_rate)
    
#     lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#     filtered_ecg_dict = {name: filtered_ecg_data[i] for i, name in enumerate(lead_names)}

#     # 2. Transformar a VCG y obtener VM
#     vcg_dict = _ecg_to_vcg(filtered_ecg_dict)
    
#     # 3. Encontrar picos R
#     r_peaks = _find_r_peaks(vcg_dict['VM'], sampling_rate)
#     if len(r_peaks) < 3:
#         print("Advertencia: Se detectaron menos de 3 latidos, no se puede continuar.")
#         return np.nan
        
#     # 4. Crear latido mediano
#     median_beat_tuple = _create_median_beat(filtered_ecg_data, r_peaks, sampling_rate)
#     if median_beat_tuple is None:
#         return np.nan
#     median_beat_data, median_r_peak = median_beat_tuple
    
#     median_ecg_dict = {name: median_beat_data[i] for i, name in enumerate(lead_names)}
    
#     # 5. Anotar el latido mediano para obtener los puntos fiduciales finales
#     median_vcg = _ecg_to_vcg(median_ecg_dict)
#     _, s_point, _, tend_point = _annotate_single_beat(median_vcg['VM'], median_r_peak, sampling_rate)

#     # 6. Calcular el TMD usando los datos del latido mediano y sus fiduciales
#     tmd_value = calculate_tmd(median_ecg_dict, s_point, tend_point)

#     return tmd_value








