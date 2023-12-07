import torch
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
loaded_model = torch.load('customer_journey_transformer.pth')
loaded_model.eval()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = loaded_model.to(device)

# Assuming you have a new dataset 'new_data.csv' in a similar format to 'df'
df = pd.read_csv('export-8.csv').head(10000)

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

# Calculate the time differences and replace padding with 0
# First, convert the string representations of lists into actual lists of strings
#df['action_dates'] = df['action_dates'].apply(ast.literal_eval)
df['action_dates'] = df['action_dates'].apply(ast.literal_eval).apply(pd.to_datetime)

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

# Convert numpy arrays to tensors
combined_features_array = np.array(df['padded_combined_features'].tolist())
X = combined_features_array[:, :-1, :]

# Convert numpy arrays to tensors
X_actions_tensor_new = torch.tensor(X[:, :, 0], dtype=torch.long).to(device)
X_time_diffs_tensor_new = torch.tensor(X[:, :, 1], dtype=torch.long).to(device)

# Model Prediction
loaded_model.eval()
with torch.no_grad():
    new_output = loaded_model((X_actions_tensor_new, X_time_diffs_tensor_new))

# Select the last timestep's predictions for each sequence
last_timestep_predictions = new_output[:, -1, :]

# Get the predicted classes (indices)
_, predicted_indices = torch.max(last_timestep_predictions, 1)

# Decode the predicted indices back to the original actions
predicted_actions = label_encoder.inverse_transform(predicted_indices.cpu().numpy())

# predicted_actions now contains the decoded textual representation
print(predicted_actions)


