
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

#------------ Dataset

vin_df = pd.read_csv("captions.csv")

vin_words = set()

for vin_caption in vin_df["caption"]:
    for vin_word in vin_caption.lower().split():
        vin_words.add(vin_word)

vin_vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + list(vin_words)

vin_word_to_id = {}
for vin_i in range(len(vin_vocab)):
    vin_word_to_id[vin_vocab[vin_i]] = vin_i

vin_id_to_word = {}
for vin_word, vin_id in vin_word_to_id.items():
    vin_id_to_word[vin_id] = vin_word

vin_vocab_size = len(vin_vocab)

#------------ Encode captions

def vin_encode(vin_caption):
    vin_ids = [1]

    for vin_word in vin_caption.lower().split():
        vin_ids.append(vin_word_to_id[vin_word])

    vin_ids.append(2)

    vin_ids = vin_ids[:20]

    while len(vin_ids) < 20:
        vin_ids.append(0)

    return vin_ids

#------------ Custom Dataset

class vin_Dataset(Dataset):
    def __init__(self):
        self.vin_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(vin_df)

    def __getitem__(self, vin_index):
        vin_name = vin_df.iloc[vin_index]["image"]
        vin_caption = vin_df.iloc[vin_index]["caption"]

        vin_image = Image.open(
            os.path.join("images", vin_name)
        ).convert("RGB")

        vin_image = self.vin_transform(vin_image)
        vin_caption = torch.tensor(vin_encode(vin_caption))

        return vin_image, vin_caption

vin_dataset = vin_Dataset()
vin_loader = DataLoader(vin_dataset, batch_size=8, shuffle=True)

#------------ CNN Image Encoder

class vin_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vin_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(128, 128)
        )

    def forward(self, vin_x):
        return self.vin_cnn(vin_x)

#------------ Caption Generator

class vin_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.vin_embedding = nn.Embedding(vin_vocab_size, 64)
        self.vin_lstm = nn.LSTM(64, 128, batch_first=True)
        self.vin_fc = nn.Linear(128, vin_vocab_size)

        self.vin_init = nn.Linear(128, 128)

    def forward(self, vin_features, vin_caption):
        vin_hidden = torch.tanh(self.vin_init(vin_features))
        vin_hidden = vin_hidden.unsqueeze(0)

        vin_cell = torch.zeros_like(vin_hidden)

        vin_input = self.vin_embedding(vin_caption[:, :-1])

        vin_output, _ = self.vin_lstm(
            vin_input, (vin_hidden, vin_cell)
        )

        return self.vin_fc(vin_output)

#------------ Discriminator

class vin_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.vin_embedding = nn.Embedding(vin_vocab_size, 64)

        self.vin_fc = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, vin_image_features, vin_caption):
        vin_text = self.vin_embedding(vin_caption)

        vin_text = vin_text.mean(dim=1)

        vin_combined = torch.cat(
            (vin_image_features, vin_text), dim=1
        )

        return self.vin_fc(vin_combined)

#------------ Initialize models

vin_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

vin_encoder = vin_Encoder().to(vin_device)
vin_generator = vin_Generator().to(vin_device)
vin_discriminator = vin_Discriminator().to(vin_device)

vin_loss = nn.BCEWithLogitsLoss()

vin_g_optimizer = torch.optim.Adam(
    list(vin_encoder.parameters()) +
    list(vin_generator.parameters()),
    lr=0.001
)

vin_d_optimizer = torch.optim.Adam(
    vin_discriminator.parameters(),
    lr=0.001
)

#------------ Training

for vin_epoch in range(5):

    for vin_images, vin_captions in vin_loader:

        vin_images = vin_images.to(vin_device)
        vin_captions = vin_captions.to(vin_device)

        vin_features = vin_encoder(vin_images)

        # Generate caption word predictions
        vin_output = vin_generator(
            vin_features, vin_captions
        )

        vin_fake_caption = torch.argmax(
            vin_output, dim=2
        )

        #------------ Train Discriminator

        vin_real_score = vin_discriminator(
            vin_features.detach(),
            vin_captions[:, 1:]
        )

        vin_fake_score = vin_discriminator(
            vin_features.detach(),
            vin_fake_caption.detach()
        )

        vin_real_label = torch.ones_like(vin_real_score)
        vin_fake_label = torch.zeros_like(vin_fake_score)

        vin_d_loss = (
            vin_loss(vin_real_score, vin_real_label) +
            vin_loss(vin_fake_score, vin_fake_label)
        )

        vin_d_optimizer.zero_grad()
        vin_d_loss.backward()
        vin_d_optimizer.step()

        #------------ Train Generator

        vin_caption_loss = F.cross_entropy(
            vin_output.reshape(-1, vin_vocab_size),
            vin_captions[:, 1:].reshape(-1),
            ignore_index=0
        )

        # Freeze discriminator parameters
        for vin_p in vin_discriminator.parameters():
            vin_p.requires_grad = False

        vin_fake_score = vin_discriminator(
            vin_features, vin_fake_caption
        )

        vin_g_loss = vin_caption_loss + vin_loss(
            vin_fake_score, vin_real_label
        )

        vin_g_optimizer.zero_grad()
        vin_g_loss.backward()
        vin_g_optimizer.step()

        for vin_p in vin_discriminator.parameters():
            vin_p.requires_grad = True

    print(
        "Epoch:", vin_epoch + 1,
        "Generator Loss:", vin_g_loss.item(),
        "Discriminator Loss:", vin_d_loss.item()
    )

#------------ Save models

torch.save(vin_encoder.state_dict(), "encoder.pth")
torch.save(vin_generator.state_dict(), "generator.pth")
torch.save(vin_discriminator.state_dict(), "discriminator.pth")

print("Models saved")

