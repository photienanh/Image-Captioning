import torch
from torch.utils.data import Dataset
import pickle
from PIL import Image
import os

class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_path, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            captions_path (string): Path to the pickled file with encoded captions.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load the dictionary of encoded captions
        with open(captions_path, 'rb') as f:
            self.captions_dict = pickle.load(f)
            
        # Create a flat list of (image_filename, caption_sequence)
        self.data = []
        for img_filename, captions_list in self.captions_dict.items():
            # Check if image file exists before adding it to the dataset
            if os.path.exists(os.path.join(self.image_dir, img_filename)):
                for caption in captions_list:
                    self.data.append((img_filename, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, caption_seq = self.data[idx]
        
        # Load image from file
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Convert caption sequence to a tensor
        caption = torch.tensor(caption_seq, dtype=torch.long)
        
        return image, caption