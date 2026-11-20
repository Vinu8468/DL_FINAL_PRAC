"""Lab 15: Image captioning with a CNN encoder and vanilla RNN decoder.

Provide captions.csv with columns image,caption and an images/ directory.
If they are absent, a small synthetic image-caption dataset is created in memory.
"""
import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms

vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_max_length = 18
vin_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

class VinVocabulary:
    def __init__(self, vin_captions):
        vin_words = set()
        for vin_caption in vin_captions:
            for vin_word in re.findall(r"[a-z]+", vin_caption.lower()):
                vin_words.add(vin_word)
        self.word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        for vin_word in sorted(vin_words):
            self.word_to_index[vin_word] = len(self.word_to_index)
        self.index_to_word = {vin_index: vin_word for vin_word, vin_index in self.word_to_index.items()}
    def encode(self, vin_caption, vin_length):
        vin_tokens = [1]
        for vin_word in re.findall(r"[a-z]+", vin_caption.lower()):
            vin_tokens.append(self.word_to_index.get(vin_word, 3))
        vin_tokens.append(2)
        vin_tokens = vin_tokens[:vin_length]
        while len(vin_tokens) < vin_length:
            vin_tokens.append(0)
        return torch.tensor(vin_tokens, dtype=torch.long)

class VinCaptionDataset(Dataset):
    def __init__(self):
        self.items = []
        if os.path.exists("captions.csv") and os.path.isdir("images"):
            vin_df = pd.read_csv("captions.csv")
            for vin_row in range(len(vin_df)):
                vin_path = os.path.join("images", str(vin_df.iloc[vin_row]["image"]))
                vin_caption = str(vin_df.iloc[vin_row]["caption"])
                self.items.append((vin_path, vin_caption, None))
            self.synthetic = False
        else:
            self.synthetic = True
            for vin_i in range(40):
                vin_image = Image.new("RGB", (64, 64), (30 + vin_i * 3 % 200, 80, 140))
                vin_draw = ImageDraw.Draw(vin_image)
                vin_draw.ellipse((15, 15, 48, 48), fill=(220, 180, 60))
                self.items.append((None, "a synthetic image with a circle", vin_image))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, vin_index):
        vin_path, vin_caption, vin_memory_image = self.items[vin_index]
        if self.synthetic:
            vin_image = vin_memory_image
        else:
            vin_image = Image.open(vin_path).convert("RGB")
        return vin_transform(vin_image), vin_caption

vin_dataset = VinCaptionDataset()
vin_captions = [vin_dataset.items[vin_i][1] for vin_i in range(len(vin_dataset))]
vin_vocab = VinVocabulary(vin_captions)

class VinCaptionCollate:
    def __call__(self, vin_batch):
        vin_images = torch.stack([vin_item[0] for vin_item in vin_batch])
        vin_captions = torch.stack([vin_vocab.encode(vin_item[1], vin_max_length) for vin_item in vin_batch])
        return vin_images, vin_captions

vin_loader = DataLoader(vin_dataset, batch_size=8, shuffle=True, collate_fn=VinCaptionCollate())

class VinCNNRNN(nn.Module):
    def __init__(self, vin_vocab_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, 128)
        )
        self.initial_state = nn.Linear(128, 128)
        self.embedding = nn.Embedding(vin_vocab_size, 64, padding_idx=0)
        self.rnn = nn.RNN(64, 128, batch_first=True)
        self.output = nn.Linear(128, vin_vocab_size)
    def forward(self, vin_images, vin_input_tokens):
        vin_features = self.encoder(vin_images)
        vin_hidden = torch.tanh(self.initial_state(vin_features)).unsqueeze(0)
        vin_embedded = self.embedding(vin_input_tokens)
        vin_rnn_output, vin_hidden = self.rnn(vin_embedded, vin_hidden)
        return self.output(vin_rnn_output)

vin_model = VinCNNRNN(len(vin_vocab.word_to_index)).to(vin_device)
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
vin_loss_function = nn.CrossEntropyLoss(ignore_index=0)
for vin_epoch in range(5):
    vin_model.train()
    vin_total_loss = 0.0
    for vin_images, vin_captions_batch in vin_loader:
        vin_images = vin_images.to(vin_device)
        vin_captions_batch = vin_captions_batch.to(vin_device)
        vin_input_tokens = vin_captions_batch[:, :-1]
        vin_target_tokens = vin_captions_batch[:, 1:]
        vin_logits = vin_model(vin_images, vin_input_tokens)
        vin_loss = vin_loss_function(vin_logits.reshape(-1, vin_logits.shape[-1]), vin_target_tokens.reshape(-1))
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()
        vin_total_loss = vin_total_loss + vin_loss.item()
    print("Epoch", vin_epoch + 1, "loss", vin_total_loss / len(vin_loader))

torch.save({"model": vin_model.state_dict(), "vocabulary": vin_vocab.word_to_index}, "15_rnn_image_caption.pth")
print("Training complete. This compact example demonstrates the CNN + vanilla RNN architecture.")
