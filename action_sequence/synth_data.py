# Example of synthetic customer journey data
customer_journeys = [
    ['website_visit', 'ad_impression', 'click', 'conversion'],
    ['email_open', 'ad_impression', 'website_visit', 'click', 'conversion'],
    ['ad_impression', 'website_visit', 'click'],
    # ... add as many sequences as needed for training
]

# Example of encoding the events into integers
event_to_token = {
    'website_visit': 0,
    'ad_impression': 1,
    'click': 2,
    'email_open': 3,
    'conversion': 4,
    'PAD': 5  # Padding token for sequences of different lengths
}

# Example of decoding tokens back into events
token_to_event = {v: k for k, v in event_to_token.items()}

# Encoding the journeys into token IDs
tokenized_journeys = [[event_to_token[event] for event in journey] for journey in customer_journeys]

from torch.nn.utils.rnn import pad_sequence
import torch

# Padding sequences to the same length
padded_sequences = pad_sequence([torch.tensor(journey) for journey in tokenized_journeys], batch_first=True, padding_value=event_to_token['PAD'])

# Creating input-target pairs
input_sequences = padded_sequences[:, :-1]  # all but the last token
target_sequences = padded_sequences[:, 1:]  # all but the first token

print(input_sequences)

import torch.nn as nn

class CustomerJourneyTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, dim_model))
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_layers=num_layers, dropout=dropout)
        self.output_layer = nn.Linear(dim_model, num_tokens)
    
    def forward(self, input_sequence):
        seq_length = input_sequence.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0)
        embedded = self.embedding(input_sequence) + self.positional_encoding[:, :seq_length]
        transformed = self.transformer(embedded, embedded)
        output = self.output_layer(transformed)
        return output

# Instantiate the model
model = CustomerJourneyTransformer(
    num_tokens=len(event_to_token), 
    dim_model=512, 
    num_heads=8, 
    num_layers=3
)

# Model summary
print(model)
