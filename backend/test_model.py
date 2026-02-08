
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ijson

# ================= CONFIG =================
MODEL_PATH = "seismic_cnn.pth"
WINDOW = 2000
SKIP_TRACES = 25000  # Train used first 25000, test on next 200
TEST_TRACES = 2000  # Larger test set for stronger statistics

# ================= MODEL DEFINITION (Must match train.py) =================
class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

# ================= LOAD DATA =================
print(f"Loading test data (skipping first {SKIP_TRACES})...")

# Load meta
meta = pd.read_csv(
    "STEAD_3C_meta.csv",
    usecols=["p_arrival_sample"],
    skiprows=range(1, SKIP_TRACES + 1), # Skip header + 5000 rows
    nrows=TEST_TRACES
).reset_index(drop=True)

print("Test meta rows:", len(meta))

# Load waveforms
waveforms = []
with open("STEAD_3C_data.json", "rb") as f:
    for i, (k, v) in enumerate(ijson.kvitems(f, "", use_float=True)):
        if i < SKIP_TRACES:
            continue
        if i >= SKIP_TRACES + TEST_TRACES:
            break
        waveforms.append(v)

print("Test waveforms:", len(waveforms))

# ================= HELPERS =================
def fix_length(x, T=WINDOW):
    if x.shape[1] >= T:
        return x[:, :T]
    return np.pad(x, ((0,0),(0,T - x.shape[1])))

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)

# ================= INFERENCE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SeismicCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: {MODEL_PATH} not found. Run train.py first.")
    exit(1)

model.eval()

correct = 0
total = 0
tp, tn, fp, fn = 0, 0, 0, 0

print("\nRunning inference...")
with torch.no_grad():
    for i in range(len(meta)):
        # Prepare input
        sample = waveforms[i]
        p = int(meta.iloc[i]["p_arrival_sample"])
        
        Z = np.array(sample["Z"], dtype=float)
        Nn = np.array(sample["N"], dtype=float)
        E = np.array(sample["E"], dtype=float)
        
        # Ground Truth check (synthetic logic from train.py)
        # In train.py: events are sliced at P-arrival, noise is before P.
        # Here we can simulate testing both "Event" and "Noise" from the same trace
        
        # Test 1: Event (Slice at P) -> Should be 1
        Zs, Ns, Es = Z[p:p+WINDOW], Nn[p:p+WINDOW], E[p:p+WINDOW]
        x_evt = np.stack([Zs, Ns, Es])
        x_evt = normalize(fix_length(x_evt))
        t_evt = torch.tensor(x_evt, dtype=torch.float32).unsqueeze(0).to(device)
        
        pred_evt = model(t_evt).item()
        label_evt = 1
        
        if pred_evt > 0.5:
            tp += 1
            correct += 1
        else:
            fn += 1
            
        # Test 2: Noise (Slice before P) -> Should be 0
        Zs, Ns, Es = Z[:p], Nn[:p], E[:p]
        x_noise = np.stack([Zs, Ns, Es])
        x_noise = normalize(fix_length(x_noise))
        t_noise = torch.tensor(x_noise, dtype=torch.float32).unsqueeze(0).to(device)
        
        pred_noise = model(t_noise).item()
        label_noise = 0
        
        if pred_noise < 0.5:
            tn += 1
            correct += 1
        else:
            fp += 1
            
        total += 2

# ================= METRICS =================
accuracy = correct / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n================ RESULTS ================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("=========================================")
