from utils.load_data import create_train_test_sets
import torch
from utils.train import train_and_test_model_binary
from fuzzy_utils.fuzzy_models import SOFIN
from torch.utils.data import DataLoader
train_dataset,test_dataset=create_train_test_sets(seed=0)
# Define batch size
batch_size = 32  # You can modify this as per your requirement
# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Define your model
sample_data, sample_label = next(iter(train_loader))
model = SOFIN(n_rules=5, input_dim=sample_data.view(sample_data.shape[0],-1).shape[-1],  output_dim=1,order=0)

accuracy = train_and_test_model_binary(model, learning_rate=0.001, train_loader=train_loader, test_loader=test_loader, epochs=10)
print("final acc=",accuracy)
#