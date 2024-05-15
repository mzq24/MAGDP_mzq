import torch
import torch.nn as nn

class TransformerCombiner(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TransformerCombiner, self).__init__()
        
        # Create the TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=1, dim_feedforward=256*2),
            num_layers
        )
        
        # Linear layer to map output to desired size
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, features):
        # features: tensor of shape (batch_size, num_features, input_size)
        
        # Permute dimensions for Transformer input
        transformer_input = features.permute(1, 0, 2)  # (num_features, batch_size, input_size)
        
        # Apply TransformerEncoder
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Permute dimensions back to original shape
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, num_features, hidden_size)
        
        # Apply linear layer
        combined_features = self.linear(transformer_output)
        
        return combined_features

if __name__ == '__main__':
    # Example usage
    batch_size = 32
    num_features = 3
    input_size = 128
    hidden_size = 128
    num_layers = 2

    # Create the TransformerCombiner
    combiner = TransformerCombiner(input_size, hidden_size, num_layers)

    # Generate random feature vectors
    features = torch.randn(batch_size, num_features, input_size)

    # Apply the TransformerCombiner
    combined_features = combiner(features)

    print('features shape', features.shape)
    print("Combined features shape:", combined_features.shape)