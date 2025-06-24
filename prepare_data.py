import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class CaptionDataset(Dataset):
    def __init__(self, features_path, captions_path):
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        with open(captions_path, 'rb') as f:
            self.captions = pickle.load(f)
        self.data = []
        for img, caps in self.captions.items():
            for cap in caps:
                self.data.append((img, cap))
        self.feature_dim = next(iter(self.features.values())).shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, cap = self.data[idx]
        feature = torch.tensor(self.features[img], dtype=torch.float32)
        caption = torch.tensor(cap, dtype=torch.long)
        return feature, caption

# Sử dụng
dataset = CaptionDataset(
    features_path='Processed Data/image_features_resnet50.pkl',
    captions_path='Processed Data/encoded_captions.pkl'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)