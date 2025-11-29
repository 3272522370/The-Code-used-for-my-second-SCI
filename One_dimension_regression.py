import os,glob,rasterio,random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
TARGET_POSITIONS = [(50,50),(60,60),(50,60)]
class GeoTiffDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, "*.tif"))
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as src:
            data_np = src.read(1).astype(float)
        # copy the array so we can zero-out target positions for the input
        data_np_mod = data_np.copy()
        labels = []
        for pos in TARGET_POSITIONS:
            r, c = pos[0], pos[1]
            labels.append(float(data_np_mod[r, c]))
            data_np_mod[r, c] = 0.0
        data_tensor = torch.from_numpy(data_np_mod).unsqueeze(0).float()
        label_tensor = torch.tensor(labels).float()
        return data_tensor, label_tensor
class SimpleNet(nn.Module):
    def __init__(self, height, width, hidden_size, out_dim):
        super().__init__()
        self.input_dim = int(height) * int(width)
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.activation = nn.ReLU()  # add Relu as activation function
        self.fc2 = nn.Linear(hidden_size, out_dim)
    def forward(self, x):
        # x expected shape: (B, 1, H, W) or (B, H, W)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
def split_dataset(dataset, test_ratio: float = 0.2, seed: int = 42):
    n = len(dataset)
    if n == 0:
        return Subset(dataset, []), Subset(dataset, [])
    indices = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    test_size = int(n * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)
def Train(train_ds, batch_size, epochs, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    # sample_x shape: (1, 112, 73) sample_y shape: (1)
    sample_x, sample_y = train_ds[0]
    _, H, W = sample_x.shape
    out_dim = int(sample_y.numel())
    hidden_size = 128
    model = SimpleNet(H, W, hidden_size, out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    #存下损失值
    losses=[]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
        if total_samples > 0:
            losses.append(running_loss/total_samples)
            print(f"Epoch {epoch+1}/{epochs} - train MAE: {losses[-1]:.6f}")
    return model, device, losses
def Test(model, test_ds, batch_size, device):
    pin_memory = True if device.type == 'cuda' else False
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    loss_fn = nn.L1Loss(reduction='mean')
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            batch_size_actual = xb.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            # collect per-sample prediction and ground-truth as Python lists
            preds_np = preds.cpu().numpy()
            yb_np = yb.cpu().numpy()
            for i in range(preds_np.shape[0]):
                predictions_list.append([preds_np[i].tolist(), yb_np[i].tolist()])
    if total_samples == 0:
        print("No samples evaluated.")
        return None, []
    test_mae = total_loss / total_samples
    print(f"Test MAE: {test_mae:.6f}")
    return test_mae, predictions_list
if __name__=="__main__":
    folder_path=os.path.join("鄱阳县2023")
    dataset = GeoTiffDataset(folder_path)
    train_ds, test_ds = split_dataset(dataset, test_ratio=0.2)
    batch_size = 12
    model, device, losses = Train(train_ds=train_ds, batch_size=batch_size, epochs=20)
    Test(model, test_ds=test_ds, batch_size=batch_size, device=device)