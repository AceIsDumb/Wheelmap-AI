import os
import pandas as pd
from PIL import Image
import pillow_heif
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Create a mapping label -> index (for classification)
        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_id']
        label_name = self.data.iloc[idx]['label']
        label = self.class_to_idx[label_name]

        # Load HEIC image
        img_path = os.path.join(self.img_dir, img_name)
        heif_file = pillow_heif.read_heif(img_path)
        image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data, "raw"
        )

        if self.transform:
            image = self.transform(image)

        return image, label
