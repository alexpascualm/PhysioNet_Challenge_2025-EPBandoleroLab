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


PROB_THRESHOLD=0.6

BATCH_SIZE=128
VCG_TRANSFORM=False
INPUT_CHANNELS=3 if VCG_TRANSFORM else 12
SEGMENTS_LENGTH=1024

R_RATIO=5

LR=1e-4
WEIGHT_DECAY=3e-4


LEAD_DROPOUT=0.3
CNN_KERNEL_SIZE=25
TRANSFORMER_N_HEAD=8
TRANSFORMER_DIM_FEEDFORWARD=1024
TRANSFORMER_DROPOUT=0.4
TRANSFORMER_NUM_LAYERS=6
FCNN_DROPOUT=0.3

ALPHA=0.25
GAMMA=2.0

# Normalization values for inference with R_RATIO=5
Inference_means=[ 0.04516214,  0.03808608, -0.00758295, -0.04170301,  0.02613188,  0.01492094, -0.04353872, -0.01552415, -0.00413191,  0.04116291,  0.06396168, 0.06180771] if not VCG_TRANSFORM else [None, None, None] 
Inference_stds=[0.31361437, 0.29698292, 0.29179248, 0.26847293, 0.26373582, 0.24906653, 0.36738571, 0.42411322, 0.50764618, 0.48428141, 0.48996164, 0.44913819] if not VCG_TRANSFORM else [None, None, None] 


################################################################################
#
# Functions to set seed and ensure reproducibility.
#
################################################################################

def set_all_seeds(seed=42):
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # Deterministic operations in cuDNN
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # For DataLoader reproducibility
    def seed_worker(worker_id):
        worker_seed=seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker

# Call the seeding function at the beginning
seed=123
worker_seed_fn=set_all_seeds(seed)



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

    records=obtain_balanced_train_dataset(data_folder, negative_to_positive_ratio=R_RATIO)
    
    num_records=len(records)

    if num_records==0:
        raise FileNotFoundError('No data were provided.')

   # Extract features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    train_data_records=[]

    # Iterate over the records
    for i in range(num_records):
        if verbose:
            width=len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        
        record=records[i]

        label=load_label(record)
        signal_data=load_signals(record)
        signal=signal_data[0]
        sampling_frequency=get_sampling_frequency(load_header(record))
        source=load_source(record)

        processed_signals=preprocess_12_lead_signal(signal, sampling_frequency, source, segments_lenght=SEGMENTS_LENGTH, vcg=VCG_TRANSFORM)
        
        # Add one entry per processed segment
        for j, processed_signal in enumerate(processed_signals):
                train_data_records.append({
                    'record': record,
                    'variant_index': j,  # index within preprocess output
                    'signal': processed_signal,
                    'label': label,
                    'probability': get_probability(load_header(record), allow_missing=True),
                    'source': source,
                    'age': load_age(record),
                    'sex': load_sex(record),
                    'sampling_frequency': sampling_frequency
                })
                
    
    # Convert list of dicts to a DataFrame
    train_df=pd.DataFrame(train_data_records)
    
    # Create model folder if missing
    os.makedirs(model_folder, exist_ok=True)

    train_and_save_model(train_df,model_folder,obtain_test_metrics=False, learning_rate=LR, lead_dropout=LEAD_DROPOUT, cnn_kernel_size=CNN_KERNEL_SIZE , transformer_n_head=TRANSFORMER_N_HEAD, transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                          transformer_dropout=TRANSFORMER_DROPOUT, transformer_num_layers=TRANSFORMER_NUM_LAYERS, fcnn_dropout=FCNN_DROPOUT, weight_decay=WEIGHT_DECAY, alpha=ALPHA, gamma=GAMMA)
    
    if verbose:
        print('Done.')
        print()



# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=ChagasClassifier(cnn_kernel_size=CNN_KERNEL_SIZE , transformer_n_head=TRANSFORMER_N_HEAD, transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
                             transformer_dropout=TRANSFORMER_DROPOUT, transformer_num_layers=TRANSFORMER_NUM_LAYERS, fcnn_dropout=FCNN_DROPOUT).to(device)
    
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pth'),weights_only=True))

    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):

    signal_data=load_signals(record)
    signal=signal_data[0]
    source=load_source(record)
    sampling_frequency=get_sampling_frequency(load_header(record))

    signal_segments=preprocess_12_lead_signal(signal,sampling_frequency, source, segments_lenght=SEGMENTS_LENGTH, vcg=VCG_TRANSFORM)                                  
    segments_tensor = np.stack(signal_segments)
    num_segments = len(segments_tensor)
    # Dummy labels (content not used during inference)
    labels = np.zeros(num_segments, dtype=bool)

   
    stats_filename = 'ecg_train_stats_for_normalization.npz'

    try:
        # Load saved normalization statistics
        stats_data = np.load(stats_filename)
        means = stats_data['mean']
        stds = stats_data['std']
    except Exception as e:
        print(f"Error loading stats file: {e}")
        print("Using default normalization values.")
        means=Inference_means
        stds=Inference_stds

    test_dataset=ECGDataset(segments_tensor, labels, train_leads_mean=means , train_leads_std=stds) 
    test_loader = DataLoader(test_dataset, batch_size=num_segments, shuffle=False)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Single-batch assumption: fetch directly
        try:
            inputs, _ = next(iter(test_loader))
        except StopIteration:
            # Handle empty segments edge case
            return False, 0.0

        inputs = inputs.to(device, non_blocking=True)
        
        # Model processes all segments at once
        outputs = model(inputs).squeeze()
        
        # Apply sigmoid to outputs
        probs = torch.sigmoid(outputs)
        
    # Compute mean probability across segments
    probability_output = probs.mean().item()
    
    binary_output = probability_output > PROB_THRESHOLD

    return binary_output, probability_output


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

###################################################################################################################
#
# Dataset, model, loss and positional encoding classes
#
####################################################################################################################

class ECGDataset(Dataset):
    def __init__(self, X, y,train_leads_mean, train_leads_std, is_train=False, lead_dropout_p=0.2):
        self.X=X
        self.y=y

        self.mean=torch.tensor(train_leads_mean, dtype=torch.float32).unsqueeze(0)
        self.std=torch.tensor(train_leads_std, dtype=torch.float32).unsqueeze(0)

        self.is_train=is_train
        self.lead_dropout_p=lead_dropout_p
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ecg_sample=torch.tensor(self.X[idx], dtype=torch.float32)

        # Normalize the ECG signal using saved train statistics
        ecg_normalized=(ecg_sample - self.mean) / (self.std + 1e-8)

        if self.is_train and self.lead_dropout_p > 0:
            # Random mask across leads for dropout augmentation
            mask=(torch.rand(ecg_normalized.shape[0]) > self.lead_dropout_p).float()
            # Ensure at least one lead remains active
            if mask.sum()==0:
                random_idx=torch.randint(0, ecg_normalized.shape[0], (1,)).item()
                mask[random_idx]=1.0
            # Apply mask across the time dimension for each lead
            ecg_normalized=ecg_normalized * mask.unsqueeze(1)
        
        ecg_transposed=ecg_normalized.permute(1, 0)
        label=torch.tensor(self.y[idx], dtype=torch.float32)

        return ecg_transposed, label
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(p=dropout)

        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2]=torch.sin(position * div_term)
        pe[:, 1::2]=torch.cos(position * div_term)
        
        # Add batch dimension to positional encoding buffer
        pe=pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x=x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ChagasClassifier(nn.Module):
    def __init__(self, cnn_kernel_size=18 , transformer_n_head=8, transformer_dim_feedforward=1024, transformer_dropout=0.2, transformer_num_layers=12, fcnn_dropout=0.4):
        super().__init__()
        # 1. CNN feature extractor
        self.cnn=nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # reduces length by factor 2
            nn.Conv1d(32, 64, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # reduces length by factor 4
            nn.Conv1d(64, 128, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), # reduces length by factor 8
            nn.Conv1d(128, 256, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2), # reduces length by factor 16
        )
        
        # For 1024 input length, CNN output sequence length ~ 1024/16 = 64 and channels=256

        # 2. Transformer components
        d_model=256  # feature dimension must match CNN channels
        self.cls_token=nn.Parameter(torch.randn(1, 1, d_model)) # [CLS] token
        self.pos_encoder=PositionalEncoding(d_model=d_model, dropout=0.1, max_len=513) # max_len greater than sequence length
        
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=transformer_n_head,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True  
        )
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # 3. Final FC classifier
        self.classifier=nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(fcnn_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(fcnn_dropout),
            nn.Linear(64, 1)
        )
      

    def forward(self, x):
        # 1. Extract features with CNN -> [batch, 256, carac_len]
        cnn_features=self.cnn(x)
        
        transformer_in=cnn_features.permute(0, 2, 1) # -> [batch, carac_len, 256]
        
        # Prepend [CLS] token to the sequence
        batch_size=x.shape[0]
        cls_tokens=self.cls_token.expand(batch_size, -1, -1) # -> [batch, 1, 256]
        transformer_in=torch.cat([cls_tokens, transformer_in], dim=1) # sequence length + 1

        # Apply positional encoding
        transformer_in=self.pos_encoder(transformer_in)

        attn_output=self.transformer(transformer_in)

        # Select only the [CLS] token output (position 0)
        cls_output=attn_output[:, 0] # -> [batch, 256]
    
        # Final classification head
        return self.classifier(cls_output)

    
class FocalLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.5, gamma=2.0):
        super().__init__()
        self.pos_weight=pos_weight.clone().detach().float()

        self.alpha=alpha
        self.gamma=gamma
        # Initialize BCEWithLogitsLoss with pos_weight
        self.bce_loss=nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

    def forward(self, inputs, targets):
        # Compute weighted BCE loss
        BCE_loss=self.bce_loss(inputs, targets)
        # Compute pt = exp(-BCE_loss)
        pt=torch.exp(-BCE_loss)

        # Dynamic alpha factor
        alpha_t=self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply Focal Loss formula
        F_loss=alpha_t * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()

    
################################################################################
#
# Training helper functions
#
################################################################################

# Get data grouped by patient/record
def get_data_by_patients(df, patient_list):
    df_subset=df[df['record'].isin(patient_list)]
    X=np.stack(df_subset['signal'].tolist(), axis=0)
    y=np.array(df_subset['label'].tolist(), dtype=bool)
    return X, y

# Training loop and model saving
def train_and_save_model(df, model_folder, obtain_test_metrics, lead_dropout=0.2, cnn_kernel_size=18 , transformer_n_head=8,
                        transformer_dim_feedforward=1024, transformer_dropout=0.2, transformer_num_layers=12, fcnn_dropout=0.4,
                        learning_rate=1e-3,weight_decay=1e-5,alpha=0.7, gamma=1.0):
    # Device selection
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("GPU Available")
    else:
        print("GPU not available")

    # Group by record to obtain unique patients
    records=df['record'].unique()

    # Assume label per patient is the first record label
    patient_labels=df.groupby('record')['label'].first()

    # Aligned lists of records and labels
    records=patient_labels.index.values
    labels=patient_labels.values

    # Split patients into train and val/test
    train_patients, valtest_patients, y_train_labels, y_valtest_labels =train_test_split(records, labels, test_size=0.3, stratify=labels, random_state=42)
    
    # Get training data
    X_train, y_train=get_data_by_patients(df, train_patients)

    # Compute normalization statistics and save to disk for inference
    train_leads_mean=np.mean(X_train, axis=(0, 1))
    train_leads_std=np.std(X_train, axis=(0, 1))

    stats_filename = 'ecg_train_stats_for_normalization.npz'
    try:
        np.savez(stats_filename, mean=train_leads_mean, std=train_leads_std)
        print(f"Saved normalization stats to '{stats_filename}'")
    except Exception as e:
        print(f"Error saving stats file: {e}")

    # Training dataset and dataloader
    train_dataset=ECGDataset(X_train, y_train, train_leads_mean, train_leads_std, is_train=True, lead_dropout_p=lead_dropout)
    train_loader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, worker_init_fn=worker_seed_fn)

    # Split validation and optional test sets
    if obtain_test_metrics:
        val_patients, test_patients, y_val_labels, y_test_labels = train_test_split(
            valtest_patients,     
            y_valtest_labels,      
            test_size=0.5,         
            stratify=y_valtest_labels,
            random_state=42
     )

        # Create test dataloader
        X_test, y_test=get_data_by_patients(df, test_patients)
        test_dataset=ECGDataset(X_test, y_test, train_leads_mean, train_leads_std)
        test_loader=DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, worker_init_fn=worker_seed_fn
        )
    else:
        val_patients=valtest_patients
        test_patients=[]

    
    # Validation dataloader
    X_val, y_val=get_data_by_patients(df, val_patients)
    val_dataset=ECGDataset(X_val, y_val, train_leads_mean, train_leads_std) 
    val_loader=DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=False,num_workers=4, worker_init_fn=worker_seed_fn)
    

    # Model, optimizer, scheduler setup
    num_epochs=50
    model=ChagasClassifier(cnn_kernel_size=cnn_kernel_size, transformer_n_head=transformer_n_head, transformer_dim_feedforward=transformer_dim_feedforward,
                            transformer_dropout=transformer_dropout, transformer_num_layers=transformer_num_layers, fcnn_dropout=fcnn_dropout).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Compute positive class weight and loss function
    pos_count=np.sum(y_train==1)
    neg_count=np.sum(y_train==0)
    pos_weight=neg_count / pos_count
    pos_weight=torch.tensor([pos_weight], device=device)  
    criterion=FocalLoss(pos_weight, alpha=alpha, gamma=gamma) 
    
    # Early stopping and overfitting monitoring
    best_auprc_score=0
    patience=10
    epochs_no_improve=0

    epochs_overfitting=0
    overfitting_threshold_challenge_score = 0.15 
    overfitting_threshold_auprc = 0.10
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_outputs = []
        epoch_labels = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs=model(inputs).squeeze()
            loss=criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Store probabilities for metrics
            epoch_outputs.append(outputs.detach())
            epoch_labels.append(labels.detach())
        

        all_outputs = torch.cat(epoch_outputs)
        all_labels = torch.cat(epoch_labels)
        all_probs = torch.sigmoid(all_outputs)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        train_probs_np = all_probs.cpu().numpy()
        train_labels_np = all_labels.cpu().numpy()

        # Compute training metrics
        train_pred_labels = train_probs_np > PROB_THRESHOLD
        train_challenge_score = compute_challenge_score(train_labels_np, train_probs_np)
        train_auc, train_auprc = compute_auc(train_labels_np, train_probs_np)
        train_accuracy = compute_accuracy(train_labels_np, train_pred_labels)
        train_f1 = compute_f_measure(train_labels_np, train_pred_labels)
        
        # Validation step
        model.eval()

        running_val_loss = 0.0
        epoch_val_outputs = []
        epoch_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)   
                epoch_val_outputs.append(outputs)
                epoch_val_labels.append(labels)

        all_val_outputs = torch.cat(epoch_val_outputs)
        all_val_labels = torch.cat(epoch_val_labels)
        all_val_probs = torch.sigmoid(all_val_outputs)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        val_probs_np = all_val_probs.cpu().numpy()
        val_labels_np = all_val_labels.cpu().numpy()

        val_pred_labels = val_probs_np > PROB_THRESHOLD
        val_challenge_score = compute_challenge_score(val_labels_np, val_probs_np)
        val_auc, val_auprc = compute_auc(val_labels_np, val_probs_np)
        val_accuracy = compute_accuracy(val_labels_np, val_pred_labels)
        val_f1 = compute_f_measure(val_labels_np, val_pred_labels)
        
       # Print epoch metrics
        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {epoch_train_loss:.4f}, Challenge Score: {train_challenge_score:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, AUPRC: {train_auprc:.4f}")
        print(f"  Val   - Loss: {epoch_val_loss:.4f}, Challenge Score: {val_challenge_score:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")
        
        # Save best model by validation AUPRC
        if val_auprc > best_auprc_score:
            print("New best model found, saving...")
            best_auprc_score=val_auprc
            torch.save(model.state_dict(), os.path.join(model_folder, 'best_model.pth'))
            epochs_no_improve=0
        else:
            epochs_no_improve +=1
            if epochs_no_improve >=patience:
                print("Early stopping triggered")
                break
        
        # Scheduler step
        scheduler.step()
        
        # Overfitting detection
        if ((train_challenge_score - val_challenge_score) > overfitting_threshold_challenge_score) or ((train_auprc - val_auprc) > overfitting_threshold_auprc):  
            print("Warning: Potential overfitting detected (Train Challenge metrics significantly higher than Val)")
            epochs_overfitting +=1
            if epochs_overfitting >=5:
                print("Overfitting detected for 5 consecutive epochs, stopping training.")
                break
        else:
            epochs_overfitting=0

    # Final test evaluation if requested
    if obtain_test_metrics:
        model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pth'),weights_only=True))
        model.eval()

        running_test_loss = 0.0
        epoch_test_outputs = []
        epoch_test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                running_test_loss += loss.item() * inputs.size(0)
                
                epoch_test_outputs.append(outputs)
                epoch_test_labels.append(labels)

        all_test_outputs = torch.cat(epoch_test_outputs)
        all_test_labels = torch.cat(epoch_test_labels)

        all_test_probs = torch.sigmoid(all_test_outputs)

        epoch_test_loss = running_test_loss / len(test_loader.dataset)

        test_probs_np = all_test_probs.cpu().numpy()
        test_labels_np = all_test_labels.cpu().numpy()

        test_pred_labels = test_probs_np > PROB_THRESHOLD
        test_challenge_score = compute_challenge_score(test_labels_np, test_probs_np)
        test_auc, test_auprc = compute_auc(test_labels_np, test_probs_np)
        test_accuracy = compute_accuracy(test_labels_np, test_pred_labels)
        test_f1 = compute_f_measure(test_labels_np, test_pred_labels)

                
        # Print test metrics
        print("Test inference")
        print(f"  Test - Loss: {epoch_test_loss:.4f}, Challenge Score: {test_challenge_score:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f},  AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}")

################################################################################
#
# Preprocessing and segmentation utilities
#
################################################################################

def Zero_pad_leads(arr, target_length=1024):
    X, Y=arr.shape
    padded_array=np.zeros((target_length, Y))
    
    for col in range(Y):
        col_data=arr[:, col]
        length=len(col_data)
        
        if length < target_length:
            pad_before=(target_length - length) // 2
            pad_after=target_length - length - pad_before
            padded_array[:, col]=np.pad(col_data, (pad_before, pad_after), mode='constant')
        else:
            padded_array[:, col]=col_data[:target_length]

    return padded_array

def adjust_length_ecg_2048(arr):
    target_length=2048
    X, Y=arr.shape
    signal1=np.zeros((target_length, Y))
    signal2=np.zeros((target_length, Y))
    signals=list()
    
    for col in range(Y):
        col_data=arr[:, col]
        length=len(col_data)
        
        if length < target_length:
            signal1=Zero_pad_leads(arr,target_length=2048)
            break

        elif length > 1.5*target_length:
            signal1[:, col]=col_data[:target_length]
            signal2[:, col]=col_data[-target_length:]

        else:
            signal1[:, col]=col_data[:target_length]
            signal2=0
            
    signals.append(signal1)
    if isinstance(signal2,np.ndarray):
        signals.append(signal2)
    return signals

def adjust_length_ecg_1024(arr):
    target_length=1024
    X, Y=arr.shape
    signal1=np.zeros((target_length, Y))
    signal2=np.zeros((target_length, Y))
    signal3=np.zeros((target_length, Y))
    signal4=np.zeros((target_length, Y))
    signals=list()
    
    for col in range(Y):
        col_data=arr[:, col]
        length=len(col_data)
        
        if length < target_length:
            signal1=Zero_pad_leads(arr,target_length=1024)
            break

        elif length >=4*target_length:
            signal1[:, col]=col_data[:target_length]
            signal2[:, col]=col_data[target_length:2*target_length]
            signal3[:, col]=col_data[2*target_length:3*target_length]
            signal4[:, col]=col_data[3*target_length:4*target_length]

        elif length >=3*target_length:
            signal1[:, col]=col_data[:target_length]
            signal2[:, col]=col_data[target_length:2*target_length]
            signal3[:, col]=col_data[2*target_length:3*target_length]
            signal4=0

        elif length > 1.5*target_length:
            signal1[:, col]=col_data[:target_length]
            signal2[:, col]=col_data[-target_length:]
            signal3=0
            signal4=0
        
        else:
            signal1[:, col]=col_data[:target_length]
            signal2=0
            signal3=0
            signal4=0

    signals.append(signal1) 
    
    if isinstance(signal4,np.ndarray):
        signals.extend([signal2, signal3, signal4])
    elif isinstance(signal3,np.ndarray):
        signals.extend([signal2, signal3])
    elif isinstance(signal2,np.ndarray):
        signals.append(signal2)

    return signals


def remove_baseline_wander(signal, factor=101):
    """
    Apply median filter to remove baseline wander
    """
    y=medfilt(signal, kernel_size=factor)
    filt_signal=signal - y
    return filt_signal


def wavelet_filter(input_signal, wavelet='coif4', level=7):
    """
    Filter a signal using a wavelet decomposition and keep approximation coefficients only.
    """
    coeffs=pywt.wavedec(input_signal, wavelet, level=level)
    for i in range(1, len(coeffs)):
        coeffs[i]=np.zeros_like(coeffs[i])
    filtered=pywt.waverec(coeffs, wavelet)
    filtered=filtered[:len(input_signal)]
    return filtered


def filter_median_wavelet(ecg_signal, factor=101, level=2, wavelet='coif4'):
    # Remove baseline and then apply wavelet low-frequency reconstruction
    ecg_signal=remove_baseline_wander(signal=ecg_signal, factor=factor)
    filtered_signal=wavelet_filter(ecg_signal, wavelet=wavelet, level=level)
    return filtered_signal



# Convert 12-lead ECG into 3-channel vectorcardiogram (VCG)
def ecg_to_vcg(ecg, tr='dower'):
    """
    Convert a 12-lead ECG into a 3-channel VCG using Dower or Kors transform.
    """
    if tr=='dower':
        T=np.array([[-0.172, -0.074,  0.122,  0.231, 0.239, 0.194,  0.156, -0.010],
                      [ 0.057, -0.019, -0.106, -0.022, 0.041, 0.048, -0.227,  0.887],
                      [-0.229, -0.310, -0.246, -0.063, 0.055, 0.108,  0.022,  0.102]])
    elif tr=='kors':
        T=np.array([[-0.13, 0.05, -0.01, 0.14, 0.06, 0.54, 0.38, -0.07],
                      [ 0.06, -0.02, -0.05, 0.06, -0.17, 0.13, -0.07,  0.93],
                      [-0.43, -0.06, -0.14, -0.20, -0.11, 0.31,  0.11, -0.23]])
    
    ecg_1=ecg[:, 6:] # V1-V6 leads
    ecg_2=ecg[:, :2] # I and II leads
    ecg_red=np.concatenate([ecg_1, ecg_2], axis=1)  # Leads reorder

    ecg_red=ecg_red.T
    vcg=np.matmul(T, ecg_red).T

    return vcg


def standardize_ecg_signal(signal: np.ndarray, sampling_frequency: int, source: str) -> np.ndarray:
    """
    Standardize ECG to a consistent lead ordering and resample to 400 Hz if needed.
    """
    if source=="PTB-XL":
        # Swap aVR and aVL to match the expected order
        signal[3, :], signal[4, :]=signal[4, :].copy(), signal[3, :].copy()

    if sampling_frequency !=400:
        signal=resample_poly(signal, 400, sampling_frequency, axis=0)

    return signal


def signal_to_model_input(padded_ecg, vcg=True, filter=True):
    """
    Process a single ECG segment: optional filtering and optional VCG transform.
    """
    if filter:
        filtered=np.zeros_like(padded_ecg)
        for lead_idx in range(padded_ecg.shape[1]):         
            filtered[:, lead_idx]=filter_median_wavelet(padded_ecg[:, lead_idx])       
    else:
        filtered=padded_ecg

    if vcg:
        filtered=ecg_to_vcg(filtered)

    return filtered


def preprocess_12_lead_signal(all_lead_signal, sampling_frequency, source, segments_lenght, vcg):
    """
    Standardize ECG, segment by desired length and optionally convert to VCG.
    Returns a list of processed segments.
    """
    # Standardize frequency and lead ordering
    all_lead_signal=standardize_ecg_signal(all_lead_signal, sampling_frequency, source)

    # Segment according to requested length
    if segments_lenght==2048:
        signal_segments=adjust_length_ecg_2048(all_lead_signal)
    elif segments_lenght==1024:
        signal_segments=adjust_length_ecg_1024(all_lead_signal)


    # Process each segment (filter + optional VCG)
    processed_segments=[
        signal_to_model_input(segment, vcg=vcg, filter=True)
        for segment in signal_segments
    ]
    return processed_segments

################################################################################
#
# Functions to build a balanced training dataset across available records
#
################################################################################

# Group ages into 5-year bins
def age_group(age):
    return (age // 5) * 5

def obtain_balanced_train_dataset(path, negative_to_positive_ratio=1.0):
    """
    Select positive and negative records from the dataset with approximately the requested
    negative-to-positive ratio while matching age-sex distribution.
    """
    # Collect all records
    records=find_records(path, '.hea')
    for i in range(len(records)):
        records[i]=os.path.join(path, records[i])
    
    # Gather positive records and their age/sex
    positive_records=[]
    age_sex_distribution=[]
    for rec in records:
        if load_label(rec)==1:
            head=load_header(rec)
            age=get_age(head)
            sex=get_sex(head)
            positive_records.append(rec)
            age_sex_distribution.append((age_group(age), sex))
    
    num_positives=len(positive_records)
    if num_positives==0:
        raise ValueError("No positive records found in the dataset")
    
    # Distribution of positives by (age_bin, sex)
    positive_distribution=Counter(age_sex_distribution)
    
    # Candidate negatives
    negative_candidates=[rec for rec in records if load_label(rec)==0]
    
    # Group negatives by the same (age_bin, sex) combinations present among positives
    negative_by_combination=defaultdict(list)
    for rec in negative_candidates:
        head=load_header(rec)
        age=get_age(head)
        sex=get_sex(head)
        comb=(age_group(age), sex)
        if comb in positive_distribution:
            negative_by_combination[comb].append(rec)
    
    # Shuffle negatives inside each group
    for comb in negative_by_combination:
        random.shuffle(negative_by_combination[comb])
    
    # Desired total negatives
    total_desired_negatives=math.ceil(negative_to_positive_ratio * num_positives)
    
    selected_negatives=[]
    selected_counts={comb: 0 for comb in positive_distribution}
    
    # Greedy selection to balance groups proportionally
    while len(selected_negatives) < total_desired_negatives and any(negative_by_combination[comb] for comb in positive_distribution):
        min_ratio=float('inf')
        best_comb=None
        for comb in positive_distribution:
            if negative_by_combination[comb]:
                desired=negative_to_positive_ratio * positive_distribution[comb]
                ratio_current=selected_counts[comb] / desired if desired > 0 else float('inf')
                if ratio_current < min_ratio:
                    min_ratio=ratio_current
                    best_comb=comb
        
        if best_comb is None:
            break
        
        neg_rec=negative_by_combination[best_comb].pop()
        selected_negatives.append(neg_rec)
        selected_counts[best_comb] +=1
    
    return positive_records + selected_negatives
