#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os
import pandas as pd

from helper_code import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from collections import Counter, defaultdict
import math
import random
from scipy.signal import resample_poly, medfilt
import pywt 


PROB_THRESHOLD = 0.6

BATCH_SIZE = 128
VCG_TRANSFORM = False
INPUT_CHANNELS = 3 if VCG_TRANSFORM else 12
SEGMENTS_LENGTH = 1024  # Cambia a 2048 si quieres usar segmentos de 2048 muestras

R_RATIO = 1

LR = 1e-5
WEIGHT_DECAY = 1e-4


LEAD_DROPOUT=0.2
CNN_KERNEL_SIZE= 18
TRANSFORMER_N_HEAD=8
TRANSFORMER_DIM_FEEDFORWARD=1024
TRANSFORMER_DROPOUT= 0.2
TRANSFORMER_NUM_LAYERS =12

ALPHA = 0.7
GAMMA = 1.0

Inference_means = [0.04075131,  0.03074186, -0.01107513, -0.03587413,  0.02576222,  0.00963566, -0.03731342, -0.0152486,  -0.00810018,  0.03289378,  0.05537905,  0.05517284] if not VCG_TRANSFORM else [0.04414546, 0.02123413, 0.02622889] 
Inference_stds = [0.28388433, 0.28506747, 0.27446247, 0.24950249, 0.23966872, 0.2410652, 0.3456556,  0.42599634, 0.51099372, 0.4868435,  0.47066601 ,0.42587815] if not VCG_TRANSFORM else [0.28388433, 0.28506747, 0.27446247]

# BATCH_SIZE = [64, 128, 256]
# VCG_TRANSFORM = [True, False]
# INPUT_CHANNELS = 3 if VCG_TRANSFORM else 12

# SEGMENTS_LENGTH = [1024, 2048]  


# R_RATIO = [1, 3, 5]

# LR = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]
# WEIGHT_DECAY = [0.0, 1e-5, 5e-5, 1e-4]

# LEAD_DROPOUT= [0.0, 0.1, 0.2, 0.3]
# CNN_KERNEL_SIZE= [11, 17, 25, 35]
# TRANSFORMER_N_HEAD= [4, 8, 12, 16]
# TRANSFORMER_DIM_FEEDFORWARD=[512, 1024, 2048]
# TRANSFORMER_DROPOUT= [0.1, 0.2, 0.3]
# TRANSFORMER_NUM_LAYERS = [2, 4, 6, 8, 12]

# ALPHA = [0.25, 0.5, 0.7, 0.75]
# GAMMA = [1.0, 2.0, 3.0]

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

    records = obtain_balanced_train_dataset(data_folder, negative_to_positive_ratio=R_RATIO)
    
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

        processed_signals = preprocess_12_lead_signal(signal, sampling_frequency, source, segments_lenght = SEGMENTS_LENGTH, vcg = VCG_TRANSFORM)
        
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

    train_and_save_model(train_df,model_folder,obtain_test_metrics = True, learning_rate=LR, lead_dropout= LEAD_DROPOUT, cnn_kernel_size=CNN_KERNEL_SIZE , transformer_n_head=TRANSFORMER_N_HEAD, transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                          transformer_dropout= TRANSFORMER_DROPOUT, transformer_num_layers =TRANSFORMER_NUM_LAYERS, weight_decay=WEIGHT_DECAY, alpha=ALPHA, gamma=GAMMA)
    
    if verbose:
        print('Done.')
        print()



# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChagasClassifier(cnn_kernel_size=CNN_KERNEL_SIZE , transformer_n_head=TRANSFORMER_N_HEAD, transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                             transformer_dropout= TRANSFORMER_DROPOUT, transformer_num_layers =TRANSFORMER_NUM_LAYERS).to(device)
    
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

    signal_segments = preprocess_12_lead_signal(signal,sampling_frequency, source, segments_lenght= SEGMENTS_LENGTH, vcg= VCG_TRANSFORM) 
                                    
    segments_tensor = np.stack(signal_segments)
    labels = np.repeat(label, len(signal_segments))

    means = Inference_means
    stds = Inference_stds

    test_dataset = ECGDataset(segments_tensor, labels, train_leads_mean = means , train_leads_std= stds) 
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
# TODO Arquitecura del modelo
#
################################################################################


# 1. Clase Dataset con preprocesamiento
class ECGDataset(Dataset):
    def __init__(self, X, y,train_leads_mean, train_leads_std, is_train=False, lead_dropout_p=0.2):
        self.X = X
        self.y = y

        self.mean = torch.tensor(train_leads_mean, dtype=torch.float32).unsqueeze(0)
        self.std = torch.tensor(train_leads_std, dtype=torch.float32).unsqueeze(0)

        self.is_train = is_train
        self.lead_dropout_p = lead_dropout_p
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ecg_sample = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_train and self.lead_dropout_p > 0:
            # Crear una máscara aleatoria para las derivaciones
            # torch.rand(ecg_sample.shape[0]) -> crea un tensor de 12 valores aleatorios [0, 1)
            # > self.lead_dropout_p -> lo convierte en booleano (True si queremos MANTENER, False si queremos ELIMINAR)
            mask = (torch.rand(ecg_sample.shape[0]) > self.lead_dropout_p).float()
            
            # Para evitar que se pongan a cero TODAS las derivaciones (caso poco probable pero posible)
            # nos aseguramos de que al menos una derivación quede activa.
            if mask.sum() == 0:
                # Si todas son cero, escogemos una al azar para mantenerla
                random_idx = torch.randint(0, ecg_sample.shape[0], (1,)).item()
                mask[random_idx] = 1.0

            # Aplicar la máscara. unsqueeze(1) añade la dimensión de la secuencia para el broadcasting.
            # mask tiene forma (12) -> unsqueeze(1) -> (12, 1)
            # ecg_sample tiene forma (12, 1024)
            # La máscara se multiplica a lo largo de toda la secuencia para cada derivación.
            ecg_sample = ecg_sample * mask.unsqueeze(1)


        ecg_normalized = (ecg_sample - self.mean) / (self.std + 1e-8)
        ecg_transposed = ecg_normalized.permute(1, 0)
        label = torch.tensor(self.y[idx], dtype=torch.float32)

        return ecg_transposed, label
    
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
    def __init__(self, cnn_kernel_size=18 , transformer_n_head=8, transformer_dim_feedforward=1024, transformer_dropout= 0.2, transformer_num_layers =12):
        super().__init__()

        # 1. Extractor de características CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 32, kernel_size= cnn_kernel_size, padding= (cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # len -> len/2
            nn.Conv1d(32, 64, kernel_size= cnn_kernel_size, padding= (cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # len -> len/4
            nn.Conv1d(64, 128, kernel_size= cnn_kernel_size, padding= (cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4), # len -> len/16
            nn.Conv1d(128, 256, kernel_size= cnn_kernel_size, padding= (cnn_kernel_size - 1) // 2),
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

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout= 0.1, max_len=513) # max_len > carac_len
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead= transformer_n_head, # Cabezales de atención
            dim_feedforward= transformer_dim_feedforward, # Capa feedforward más pequeña
            dropout=transformer_dropout, # Dropout en la capa de atención
            batch_first=True  # ¡Muy importante para manejar las dimensiones fácilmente!
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers= transformer_num_layers)

        # 3. Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), # d_model es 256
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
def train_and_save_model(df, model_folder, obtain_test_metrics, learning_rate=1e-3, lead_dropout=0.2, cnn_kernel_size=18 , transformer_n_head=8, transformer_dim_feedforward=1024, transformer_dropout= 0.2, transformer_num_layers =12, weight_decay=1e-5,alpha=0.7, gamma=1.0):
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

    X_train, y_train = get_data_by_patients(df, train_patients)
    train_leads_mean = np.mean(X_train, axis=(0, 1))
    train_leads_std = np.std(X_train, axis=(0, 1))

    # print(f"Train leads mean: {train_leads_mean}"
    #       f"\nTrain leads std: {train_leads_std}")
    train_dataset = ECGDataset(X_train, y_train, train_leads_mean, train_leads_std, is_train=True, lead_dropout_p= lead_dropout)

    if obtain_test_metrics:
        val_patients, test_patients = train_test_split(
            valtest_patients, test_size=0.5, random_state=42
        )

        X_test, y_test = get_data_by_patients(df, test_patients)
        test_dataset = ECGDataset(X_test, y_test, train_leads_mean, train_leads_std)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, worker_init_fn=worker_seed_fn
        )
    else:
        val_patients = valtest_patients
        test_patients = []

    
    
    X_val, y_val = get_data_by_patients(df, val_patients)

    # Crear datasets
    val_dataset = ECGDataset(X_val, y_val, train_leads_mean, train_leads_std)
        
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True,num_workers=4, worker_init_fn=worker_seed_fn)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE,shuffle=False,num_workers=4, worker_init_fn=worker_seed_fn)
    
    
    model = ChagasClassifier(cnn_kernel_size= cnn_kernel_size, transformer_n_head= transformer_n_head, transformer_dim_feedforward= transformer_dim_feedforward, transformer_dropout= transformer_dropout, transformer_num_layers =transformer_num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / pos_count

    pos_weight = torch.tensor([pos_weight], device=device)  
    
    criterion = FocalLoss(pos_weight, alpha=alpha, gamma=gamma) 
    
    best_auprc_score = 0
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
        
        # Guardar el mejor modelo basado en el val_auprc de validación
        if val_auprc > best_auprc_score:
            print("New best model found, saving...")
            best_auprc_score = val_auprc
            torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        
        # # Ajustar el learning rate basado en el val_auprc de validación
        scheduler.step(val_auprc)
        
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
            signal1 = Zero_pad_leads(arr,target_length=2048)
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
            signal1 = Zero_pad_leads(arr,target_length=1024)
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
def remove_baseline_wander(signal, factor=101):
    # Apply median filter
    y = medfilt(signal, kernel_size=factor)
    # Subtract baseline wander
    filt_signal = signal - y
    return filt_signal


def wavelet_filter(senal_entrada, wavelet='coif4', nivel=7):
    """
    Filtra una señal utilizando transformada Wavelet.

    Args:
        senal_entrada (array-like): Señal de entrada a filtrar.
        wavelet (str): Familia de wavelet a usar (por defecto 'coif4').
        nivel (int): Número de niveles de descomposición (por defecto 7).
    
    Returns:
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


def filter_median_wavelet(ecg_signal, factor=101, level=2, wavelet='coif4'):
    # First remove baseline wander and then filter with wavelets
    ecg_signal = remove_baseline_wander(signal = ecg_signal, factor = factor)
    filtered_signal = wavelet_filter(ecg_signal, wavelet = wavelet, nivel=level)
    return filtered_signal



# VCG to ECG
def ecg_to_vcg(ecg, tr='dower'):
    """
    Convierte una señal de ECG de 12 derivaciones en una representación VCG de 3 canales 
    usando la transformación de Dower o Kors.

    Args:
    ecg : np.ndarray
        Array de forma (N, 12) donde N es el número de muestras y 12 las derivaciones del ECG.
    tr : str, optional
        Tipo de transformación a utilizar: 'dower' (por defecto) o 'kors'.

    Returns:
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

    Args:
    signal : np.ndarray
        Señal ECG con forma (N, 12), donde N es el número de muestras.
    sampling_frequency : int
        Frecuencia de muestreo original de la señal.
    source : str
        Nombre de la base de datos fuente (ej. 'PTB-XL', 'SamiTrop', 'Code-15').

    Returns:
    np.ndarray
        Señal ECG con derivaciones ordenadas y muestreada a 400 Hz.
    """
    if source == "PTB-XL":
        # Intercambiar aVR y aVL para que coincidan con el orden estándar
        signal[3, :], signal[4, :] = signal[4, :].copy(), signal[3, :].copy()

    if sampling_frequency != 400:
        signal = resample_poly(signal, 400, sampling_frequency, axis=0)

    return signal


def signal_to_model_input(padded_ecg, vcg=True, filter=True):
    """
    Procesa un segmento de señal ECG aplicando filtrado, normalización y transformación opcional a VCG.

    Args:
    padded_ecg : np.ndarray
        Segmento ECG de forma (N, 12).
    vcg : bool, optional
        Si True, convierte la señal a representación VCG. Por defecto es True.
    filter : bool, optional
        Si True, aplica un filtro. Por defecto es False.
   
    Returns:
    np.ndarray
        Segmento procesado, ya sea como ECG (N, 12) o VCG (N, 3).
    """
    if filter:
        # Placeholder para un filtro real
        filtered = np.zeros_like(padded_ecg)
        for lead_idx in range(padded_ecg.shape[1]):         
            filtered[:, lead_idx] = filter_median_wavelet(padded_ecg[:, lead_idx])       
    else:
        filtered = padded_ecg

    

    if vcg:
        filtered = ecg_to_vcg(filtered)

    return filtered

def preprocess_12_lead_signal(all_lead_signal, sampling_frequency, source, segments_lenght, vcg):
    """
    Preprocesa una señal ECG de 12 derivaciones: estandariza frecuencia y orden, 
    la segmenta y la convierte a VCG si se desea.

    Args:
    all_lead_signal : np.ndarray
        Señal ECG completa de forma (N, 12), donde N es el número de muestras.
    sampling_frequency : int
        Frecuencia de muestreo original de la señal.
    source : str
        Nombre de la base de datos fuente (ej. 'PTB-XL', 'SamiTrop', 'Code-15').

    Returns:
    list of np.ndarray
        Lista de segmentos (N,12) procesados, transformados según configuración.
    """
    # # Paso 0: Estandarizar señal
    # all_lead_signal = standardize_ecg_signal(all_lead_signal, sampling_frequency, source)

    # # Paso 1: Procesar cada segmento (VCG, normalización, etc.)
    # segments = signal_to_model_input(all_lead_signal, vcg= vcg, filter=True)
         
    # # Paso 2: Segmentar según longitud deseada
    # if segments_lenght == 2048:
    #     signal_segments = adjust_length_ecg_2048(segments)
    # elif segments_lenght == 1024:
    #     signal_segments = adjust_length_ecg_1024(segments)

    
     # Paso 0: Estandarizar señal
    all_lead_signal = standardize_ecg_signal(all_lead_signal, sampling_frequency, source)

    # Paso 1: Segmentar según longitud deseada
    if SEGMENTS_LENGTH == 2048:
        signal_segments = adjust_length_ecg_2048(all_lead_signal)
    elif SEGMENTS_LENGTH == 1024:
        signal_segments = adjust_length_ecg_1024(all_lead_signal)


    # Paso 2: Procesar cada segmento (VCG, normalización, etc.)
    processed_segments = [
        signal_to_model_input(segment, vcg=vcg, filter=True)
        for segment in signal_segments
    ]
    return processed_segments

    # return signal_segments

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