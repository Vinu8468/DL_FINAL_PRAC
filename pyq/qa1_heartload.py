import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("heart.csv")

# 2. Separate features and target
X = df.drop("target", axis=1).values
y = df["target"].values

# 3. Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 4. Custom Dataset
class HeartDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# 5. Split into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# 6. Create Dataset objects
train_dataset = HeartDataset(X_train, y_train)
test_dataset = HeartDataset(X_test, y_test)

# 7. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 8. Print sizes
print("Training dataset size:", len(train_dataset))
print("Testing dataset size:", len(test_dataset))

print("Training batches:", len(train_loader))
print("Testing batches:", len(test_loader))

##############################################
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

vin_data = pd.read_csv("heart.csv")

vin_x = torch.tensor(vin_data.drop("target", axis=1).values, dtype=torch.float32)
vin_y = torch.tensor(vin_data["target"].values, dtype=torch.float32)

class HeartDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

vin_x_train, vin_x_test, vin_y_train, vin_y_test = train_test_split(
    vin_x, vin_y, test_size=0.3, random_state=42
)

vin_train = HeartDataset(vin_x_train, vin_y_train)
vin_test = HeartDataset(vin_x_test, vin_y_test)

vin_train_loader = DataLoader(vin_train, batch_size=32, shuffle=True)
vin_test_loader = DataLoader(vin_test, batch_size=32, shuffle=False)

print("Train size:", len(vin_train))
print("Test size:", len(vin_test))
print("Train batches:", len(vin_train_loader))
print("Test batches:", len(vin_test_loader))
