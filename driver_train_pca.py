################# Code Notes #################

'''
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

'''
################# /Code Notes #################

import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.optim.lr_scheduler import OneCycleLR

# Data
df = pd.read_csv('export-8.csv').head(5000)

# Convert actions column into a list instead of a representation of a list
for i, sublist_str in enumerate(df['actions']):
    try:
        sublist = ast.literal_eval(sublist_str)
        df.at[i, 'actions'] = sublist
    except ValueError:
        pass

# Flatten the list of actions and add a dummy action for padding
all_actions = ['<PAD>'] + [action for sublist in df['actions'] for action in sublist if action != -1]

# Fit a label encoder to all possible actions including the dummy padding action
label_encoder = LabelEncoder()
label_encoder.fit(all_actions)

# Apply encoding to each action sequence and ensure padding is 0
df['encoded_actions'] = df['actions'].apply(lambda x: [0 if action == -1 else label_encoder.transform([action])[0] for action in x])

# Convert the string representations of lists into actual lists of strings
def convert_to_dates(list_str):
    try:
        date_list = ast.literal_eval(list_str)
        return [pd.to_datetime(date) for date in date_list]
    except (ValueError, SyntaxError):
        return []

df['action_dates'] = df['action_dates'].apply(convert_to_dates)

# Calculate the time differences
df['time_diffs'] = df['action_dates'].apply(
    lambda x: [0] + [(pd.to_datetime(x[i]) - pd.to_datetime(x[i-1])).days for i in range(1, len(x))]
)

# Padding the sequences
max_combined_length = df[['encoded_actions', 'time_diffs']].applymap(len).max().max()
df['padded_combined_features'] = df.apply(
    lambda row: pad_sequences([list(zip(row['encoded_actions'], row['time_diffs']))],
                              maxlen=max_combined_length, padding='post', value=(0, 0))[0],
    axis=1
)

# Split data to x and y
combined_features_array = np.array(df['padded_combined_features'].tolist())
X = combined_features_array[:, :-1, :]
y = combined_features_array[:, -1, 0]  # [:, -1, :] <- predict both actions and time; [:, -1, 0] <- predict only actions

# Define the number of tokens and time differences
num_tokens = len(label_encoder.classes_)
num_time_diffs = df['time_diffs'].explode().max() + 1

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer Next Best Driver model
class CustomerJourneyTransformer(nn.Module):
    def __init__(self, num_tokens, num_time_diffs, dim_model, num_heads, num_encoder_layers, max_combined_length, dropout=0.1):
        super().__init__()
        self.action_embedding = nn.Embedding(num_tokens, dim_model, padding_idx=0)
        self.time_diff_embedding = nn.Embedding(num_time_diffs, dim_model, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_combined_length, dim_model))
        self.layer_norm_emb = nn.LayerNorm(dim_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # PCA as a linear layer for dimensionality reduction
        self.pca = nn.Linear(dim_model, 2)

        self.output_layer = nn.Linear(2, num_tokens)  # Output layer now takes the 2D PCA output

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.action_embedding.weight, mean=0, std=0.02)
        init.normal_(self.time_diff_embedding.weight, mean=0, std=0.02)
        init.xavier_uniform_(self.output_layer.weight)
        if hasattr(self.output_layer, 'bias') and self.output_layer.bias is not None:
            init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        action_sequence, time_diff_sequence = x
        action_embeddings = self.action_embedding(action_sequence)
        time_diff_embeddings = self.time_diff_embedding(time_diff_sequence)
        embeddings = action_embeddings + time_diff_embeddings + self.positional_encoding[:, :action_sequence.size(1), :]
        embeddings = self.layer_norm_emb(embeddings)

        transformed = self.transformer_encoder(embeddings)

        # Applying PCA after embeddings
        pca_output = self.pca(transformed)

        output = self.output_layer(pca_output)
        return output

# Instantiate the model and move it to the device
model = CustomerJourneyTransformer(
    num_tokens=num_tokens, 
    num_time_diffs=num_time_diffs,
    dim_model=512, 
    num_heads=8, 
    num_encoder_layers=3,
    max_combined_length=max_combined_length,
    dropout=0.1
).to(device)

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

# Convert numpy arrays to tensors
X_actions_tensor = torch.tensor(X[:, :, 0], dtype=torch.long)
X_time_diffs_tensor = torch.tensor(X[:, :, 1], dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create the dataset
dataset = CustomerJourneyDataset(X_actions_tensor, X_time_diffs_tensor, y_tensor)

# Training settings
batch_size = 200  # Adjust batch size
num_epochs = 50
clip_value = 0.0  # Gradient clipping value; on = 1.0, off = 0.0
learning_rate = 0.1

# Schedule learning rate
max_lr = 6e-4  # Peak learning rate
div_factor = 25  # Initial learning rate will be max_lr / div_factor
final_div_factor = 1e4  # Final learning rate will be max_lr / final_div_factor
pct_start = 0.3  # 30% of the training steps will be for ramping up the learning rate

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = OneCycleLR(optimizer, 
                       max_lr=max_lr, 
                       steps_per_epoch=len(dataloader), 
                       epochs=num_epochs, 
                       div_factor=div_factor, 
                       final_div_factor=final_div_factor, 
                       pct_start=pct_start)

# Training loop with batching
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (actions, time_diffs, targets) in enumerate(dataloader):
        actions, time_diffs, targets = actions.to(device), time_diffs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Perform forward pass
        output = model((actions, time_diffs))

        # Select the last timestep's predictions for each sequence
        last_timestep_predictions = output[:, -1, :]

        loss = criterion(last_timestep_predictions, targets)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch+1}, batch {i+1}")
            break
        
        loss.backward()
        
        # Apply gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        running_loss += loss.item()

    if epoch < 10 or (epoch + 1) % 10 == 0 or torch.isnan(loss):
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / (i+1)}')

