################# Code Notes #################

'''
Training Setup:
+-----------------------------+
|                             |
|      Training Setup         |
|                             |
+--------+---------+----------+
         |         |
         v         v
+--------+----+  +---+--------+
|             |  |            |
| DataLoader  |  | Optimizer  |
| (Batching)  |  | (Adam)     |
|             |  |            |
+-------------+  +------------+
         |           |
         |           |
         v           v
+--------+----+  +---+--------+
|             |  |            |
| Loss        |  | Learning   |
| (CrossEntropy)| | Rate      |
| Criterion   |  | Scheduler  |
|             |  | (ReduceLROn|
|             |  | Plateau)   |
+-------------+  +------------+


Model Overall:
Input Sequence
      |
      v
---------------------
| Action Embedding  |
---------------------
      +
      |
---------------------
| Time Diff Embedding |
---------------------
      +
      |
---------------------
| Positional Encoding |
---------------------
      +
      |
----------------------------
| Layer Normalization (Emb) |
----------------------------
      |
      v
------------------------------------
| Transformer Encoder Layer 1      |
| - Multi-Head Attention           |
| - Feed-Forward Network           |
| - (with Dropout)                 |
------------------------------------
      |
      v
------------------------------------
| Transformer Encoder Layer 2      |
| - Multi-Head Attention           |
| - Feed-Forward Network           |
| - (with Dropout)                 |
------------------------------------
      |
      v
------------------------------------
| Transformer Encoder Layer 3      |
| - Multi-Head Attention           |
| - Feed-Forward Network           |
| - (with Dropout)                 |
------------------------------------
      |
      v
-------------------
| Linear Output   |
-------------------
      |
      v
Predicted Output Sequence

Training Loop:
+-----------------------------+
|                             |
|      Training Loop          |
|                             |
+--------+---------+----------+
         |         
         v         
+--------+----+    
|             |   
| For Each    |   
| Epoch       |   
|             |   
+-------------+   
         |       
         v       
+--------+----+  
|             | 
| For Each    | 
| Batch in    | 
| DataLoader  | 
|             | 
+-------------+ 
         |     
         v     
+--------+----+ 
|             |
| Forward Pass| 
| & Calculate | 
| Loss        | 
|             | 
+-------------+ 
         |     
         v     
+--------+----+ 
|             | 
| Backward    | 
| Pass &      | 
| Gradient    | 
| Clipping    | 
|             | 
+-------------+ 
         |     
         v     
+--------+----+ 
|             | 
| Optimize    | 
| (Step)      | 
|             | 
+-------------+ 
         |     
         v     
+--------+----+ 
|             | 
| Update LR   | 
| via         | 
| Scheduler   | 
|             | 
+-------------+ 


'''
################# /Code Notes #################

import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.optim.lr_scheduler import ReduceLROnPlateau

#############################################

# Training settings
batch_size = 250  # Adjust batch size
num_epochs = 50
clip_value = 1.0  # Gradient clipping value; on = 1.0, off = 0.0
learning_rate = 0.01

# Scheduler setup for ReduceLROnPlateau
factor = 0.05  # Factor by which the learning rate will be reduced. new_lr = lr * factor
patience = 200  # Number of epochs with no improvement after which learning rate will be reduced.
threshold = 0.10  # Threshold for measuring the new optimum, to only focus on significant changes.
cooldown = 50  # Number of epochs to wait before resuming normal operation after lr has been reduced.
min_lr = 6e-4  # A lower bound on the learning rate of all param groups.

# Model Architeture Settings
dim_model = 512 
num_heads = 4
num_encoder_layers = 2
dropout = 0.4

# Train, val size
test_size = 0.2

# Data
df = pd.read_csv('export-8.csv').head(50_000)

#############################################

# Functions

# Transformer Next Best Driver model
class CustomerJourneyTransformer(nn.Module):
    def __init__(self, 
                 num_tokens, 
                 num_time_diffs, 
                 dim_model, 
                 num_heads, 
                 num_encoder_layers, 
                 max_combined_length, 
                 dropout=0.1):
        super().__init__()
        self.action_embedding = nn.Embedding(num_tokens, dim_model, padding_idx=0)
        self.time_diff_embedding = nn.Embedding(num_time_diffs + 1, dim_model, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_combined_length, dim_model))
        self.layer_norm_emb = nn.LayerNorm(dim_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(dim_model, num_tokens)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.action_embedding.weight, mean=0, std=0.02)
        init.normal_(self.time_diff_embedding.weight, mean=0, std=0.02)

        # Initialize the weights of the output layer
        # Choose the appropriate initialization based on your model's final layer activation
        if hasattr(self, 'output_layer') and hasattr(self.output_layer, 'weight') and self.output_layer.weight is not None:
            init.xavier_uniform_(self.output_layer.weight)

        # Set biases to zero if they exist
        if hasattr(self, 'output_layer') and hasattr(self.output_layer, 'bias') and self.output_layer.bias is not None:
            init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        action_sequence, time_diff_sequence = x
        action_embeddings = self.action_embedding(action_sequence)
        time_diff_embeddings = self.time_diff_embedding(time_diff_sequence)

        embeddings = action_embeddings + time_diff_embeddings + self.positional_encoding[:, :action_sequence.size(1), :]

        embeddings = self.layer_norm_emb(embeddings)

        transformed = self.transformer_encoder(embeddings)
        transformed = self.layer_norm_emb(transformed)

        output = self.output_layer(transformed)
        return output

# Custom Dataset
class CustomerJourneyDataset(Dataset):
    def __init__(self, actions, time_diffs, targets):
        self.actions = actions
        self.time_diffs = time_diffs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.actions[idx], self.time_diffs[idx], self.targets[idx]

def convert_to_dates(list_str):
    try:
        date_list = ast.literal_eval(list_str)
        return [pd.to_datetime(date) for date in date_list]
    except (ValueError, SyntaxError):
        return []

#############################################

# Data

# Convert actions column into a list
for i, sublist_str in enumerate(df['actions']):
    try:
        sublist = ast.literal_eval(sublist_str)
        df.at[i, 'actions'] = sublist
    except ValueError:
        pass

# Limiting sequence lenght to not overly pad the data
count_threshold = int(df['actions'].apply(len).median())
print(f'Median of sequence lenght: {count_threshold}')
count_threshold = count_threshold - 1

df['actions'] = df['actions'].apply(lambda x: x[:count_threshold])

# Flatten the list of actions and add a dummy action for padding
all_actions = ['<PAD>'] + [action for sublist in df['actions'] for action in sublist if action != -1]

# Fit a label encoder to all possible actions including the dummy padding action
label_encoder = LabelEncoder()
label_encoder.fit(all_actions)

# Apply encoding to each action sequence and ensure padding is 0
df['encoded_actions'] = df['actions'].apply(lambda x: [0 if action == -1 else label_encoder.transform([action])[0] for action in x])

# Modify action dates to real list and apply threshold
df['action_dates'] = df['action_dates'].apply(convert_to_dates)
df['action_dates'] = df['action_dates'].apply(lambda x: x[:count_threshold])

# Calculate the time differences
df['time_diffs'] = df['action_dates'].apply(
    lambda x: [0] + [(pd.to_datetime(x[i]) - pd.to_datetime(x[i-1])).days for i in range(1, len(x))]
)

# Add 1 to each time difference to shift the index range
df['offset_time_diffs'] = df['time_diffs'].apply(lambda x: [time_diff + 1 for time_diff in x])

# Padding the sequences
max_combined_length = df[['encoded_actions', 'time_diffs']].applymap(len).max().max()
df['padded_combined_features'] = df.apply(
    lambda row: pad_sequences([list(zip(row['encoded_actions'], row['offset_time_diffs']))],
                              maxlen=max_combined_length, padding='post', value=(0, 0))[0],
    axis=1
)

df.to_csv('data_check.csv')

# Define the number of tokens and time differences
num_tokens = len(label_encoder.classes_)
num_time_diffs = df['time_diffs'].explode().max() + 1

# Split data to x and y
combined_features_array = np.array(df['padded_combined_features'].tolist())
X = combined_features_array[:, :-1, :]
y = combined_features_array[:, -1, 0]

del df

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of X_val: {X_val.shape}')
print(f'Shape of y_val: {y_val.shape}')

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the device
model = CustomerJourneyTransformer(
    num_tokens=num_tokens, 
    num_time_diffs=num_time_diffs,
    dim_model=dim_model, 
    num_heads=num_heads, 
    num_encoder_layers=num_encoder_layers,
    max_combined_length=max_combined_length,
    dropout=dropout
).to(device)

# Convert numpy arrays to tensors for training set
X_train_actions_tensor = torch.tensor(X_train[:, :, 0], dtype=torch.long)
X_train_time_diffs_tensor = torch.tensor(X_train[:, :, 1], dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Convert numpy arrays to tensors for validation set
X_val_actions_tensor = torch.tensor(X_val[:, :, 0], dtype=torch.long)
X_val_time_diffs_tensor = torch.tensor(X_val[:, :, 1], dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create datasets
train_dataset = CustomerJourneyDataset(X_train_actions_tensor, X_train_time_diffs_tensor, y_train_tensor)
val_dataset = CustomerJourneyDataset(X_val_actions_tensor, X_val_time_diffs_tensor, y_val_tensor)

# Create the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#### Debugging train
'''
for i, (actions, time_diffs, targets) in enumerate(train_dataloader):
    print(f"Batch {i}")
    print("Actions:", actions)
    print("Time Differences:", time_diffs)
    print("Targets:", targets)
    if i == 1:  # Just print the first two batches
        break
'''
unique, counts = np.unique(y_train, return_counts=True)
train_distribution = dict(zip(unique, counts))

unique, counts = np.unique(y_val, return_counts=True)
val_distribution = dict(zip(unique, counts))

print("Training set target distribution:", train_distribution)
print("Validation set target distribution:", val_distribution)
####

# Training loop setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                              threshold=threshold, cooldown=cooldown, min_lr=min_lr, verbose=True)

# Training loop with batching
for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (actions, time_diffs, targets) in enumerate(train_dataloader):
        actions, time_diffs, targets = actions.to(device), time_diffs.to(device), targets.to(device)

        optimizer.zero_grad()  # Clear the gradients

        # Perform forward pass
        output = model((actions, time_diffs))

        # Select the last timestep's predictions for each sequence
        last_timestep_predictions = output[:, -1, :]

        # Compute loss
        loss = criterion(last_timestep_predictions, targets)

        # Backpropagation
        loss.backward()

        # Apply gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Update model parameters
        optimizer.step()

        # Update the learning rate
        scheduler.step(running_loss / len(train_dataloader))

        running_loss += loss.item()  # Accumulate the loss

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation during validation
        for actions, time_diffs, targets in val_dataloader:
            actions, time_diffs, targets = actions.to(device), time_diffs.to(device), targets.to(device)

            # Forward pass on validation data
            output = model((actions, time_diffs))

            # Select the last timestep's predictions for each sequence
            last_timestep_predictions = output[:, -1, :]

            # Compute loss
            loss = criterion(last_timestep_predictions, targets)
            val_loss += loss.item() # Accumulate the val loss

    # Print training and validation loss
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}')

# Save the model
model_save_path = "customer_journey_transformer_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
