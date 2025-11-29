import os
import glob
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
class GeoTiffDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, "*.tif"))
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as src:
            data_np = src.read(1)
        data_tensor = torch.from_numpy(data_np).unsqueeze(0).float()
        return data_tensor
if __name__=="__main__":
    folder_path=os.path.join("F:","2023")
    dataset = GeoTiffDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(dataloader.__len__())
