"""Lab 19: Adversarial image-caption generator using PyTorch.

Input option: captions.csv (columns image,caption) plus images/ directory.
Otherwise, synthetic images and captions are used to verify that the code runs.
A Gumbel-Softmax relaxation keeps the adversarial path differentiable through word choices.
This is a compact educational model, not a state-of-the-art captioning system.
"""
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms

vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_max_length = 16
vin_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

class VinVocabulary:
    def __init__(self, vin_captions):
        self.word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        for vin_caption in vin_captions:
            for vin_word in re.findall(r"[a-z]+", vin_caption.lower()):
                if vin_word not in self.word_to_index:
                    self.word_to_index[vin_word] = len(self.word_to_index)
    def encode(self, vin_caption):
        vin_tokens = [1]
        for vin_word in re.findall(r"[a-z]+", vin_caption.lower()):
            vin_tokens.append(self.word_to_index.get(vin_word, 3))
        vin_tokens.append(2)
        vin_tokens = vin_tokens[:vin_max_length]
        while len(vin_tokens) < vin_max_length:
            vin_tokens.append(0)
        return torch.tensor(vin_tokens, dtype=torch.long)

class VinCaptionDataset(Dataset):
    def __init__(self):
        self.items = []
        if os.path.exists("captions.csv") and os.path.isdir("images"):
            vin_df = pd.read_csv("captions.csv")
            for vin_i in range(len(vin_df)):
                self.items.append((os.path.join("images", str(vin_df.iloc[vin_i]["image"])), str(vin_df.iloc[vin_i]["caption"]), None))
            self.synthetic = False
        else:
            self.synthetic = True
            for vin_i in range(32):
                vin_image = Image.new("RGB", (64, 64), (80, 100, 150))
                vin_draw = ImageDraw.Draw(vin_image)
                if vin_i % 2 == 0:
                    vin_draw.ellipse((12, 12, 52, 52), fill=(220, 70, 60))
                    vin_caption = "a red circle on a blue background"
                else:
                    vin_draw.rectangle((12, 12, 52, 52), fill=(60, 210, 100))
                    vin_caption = "a green square on a blue background"
                self.items.append((None, vin_caption, vin_image))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, vin_index):
        vin_path, vin_caption, vin_image = self.items[vin_index]
        if not self.synthetic:
            vin_image = Image.open(vin_path).convert("RGB")
        return vin_transform(vin_image), vin_vocab.encode(vin_caption)

vin_dataset = VinCaptionDataset()
vin_vocab = VinVocabulary([vin_dataset.items[vin_i][1] for vin_i in range(len(vin_dataset))])
# Recreate dataset after the vocabulary exists, so its encoder can use the final word map.
vin_dataset = VinCaptionDataset()
vin_loader = DataLoader(vin_dataset, batch_size=8, shuffle=True)

class VinCaptionGenerator(nn.Module):
    def __init__(self, vin_vocab_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, 128)
        )
        self.initial_hidden = nn.Linear(128, 128)
        self.embedding = nn.Embedding(vin_vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.output = nn.Linear(128, vin_vocab_size)
    def forward(self, vin_images, vin_tokens):
        vin_image_features = self.encoder(vin_images)
        vin_hidden = torch.tanh(self.initial_hidden(vin_image_features)).unsqueeze(0)
        vin_cell = torch.zeros_like(vin_hidden)
        vin_embedded = self.embedding(vin_tokens)
        vin_sequence, vin_state = self.lstm(vin_embedded, (vin_hidden, vin_cell))
        vin_logits = self.output(vin_sequence)
        return vin_logits, vin_image_features

class VinCaptionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128 + 64, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, vin_image_features, vin_text_features):
        vin_joint_features = torch.cat((vin_image_features, vin_text_features), dim=1)
        return self.network(vin_joint_features)

def vin_real_text_representation(vin_tokens, vin_embedding):
    vin_embedded = vin_embedding(vin_tokens)
    vin_mask = (vin_tokens != 0).float().unsqueeze(2)
    vin_sum = torch.sum(vin_embedded * vin_mask, dim=1)
    vin_count = torch.clamp(torch.sum(vin_mask, dim=1), min=1.0)
    return vin_sum / vin_count

def vin_fake_text_representation(vin_logits, vin_embedding, vin_mask_tokens):
    vin_probabilities = F.gumbel_softmax(vin_logits, tau=1.0, hard=False, dim=2)
    vin_soft_embeddings = torch.matmul(vin_probabilities, vin_embedding.weight)
    vin_mask = (vin_mask_tokens != 0).float().unsqueeze(2)
    vin_sum = torch.sum(vin_soft_embeddings * vin_mask, dim=1)
    vin_count = torch.clamp(torch.sum(vin_mask, dim=1), min=1.0)
    return vin_sum / vin_count

vin_generator = VinCaptionGenerator(len(vin_vocab.word_to_index)).to(vin_device)
vin_discriminator = VinCaptionDiscriminator().to(vin_device)
vin_generator_optimizer = torch.optim.Adam(vin_generator.parameters(), lr=0.001)
vin_discriminator_optimizer = torch.optim.Adam(vin_discriminator.parameters(), lr=0.001)
vin_adversarial_loss = nn.BCELoss()
vin_caption_loss = nn.CrossEntropyLoss(ignore_index=0)

for vin_epoch in range(4):
    for vin_images, vin_captions in vin_loader:
        vin_images = vin_images.to(vin_device)
        vin_captions = vin_captions.to(vin_device)
        vin_input_tokens = vin_captions[:, :-1]
        vin_real_tokens = vin_captions[:, 1:]
        vin_mask_tokens = vin_real_tokens

        # ---------- Train discriminator on real and generated caption representations
        vin_generator.eval()
        with torch.no_grad():
            vin_fake_logits, vin_image_features = vin_generator(vin_images, vin_input_tokens)
            vin_fake_text = vin_fake_text_representation(vin_fake_logits, vin_generator.embedding, vin_mask_tokens)
            vin_real_text = vin_real_text_representation(vin_real_tokens, vin_generator.embedding)
        vin_real_score = vin_discriminator(vin_image_features.detach(), vin_real_text.detach())
        vin_fake_score = vin_discriminator(vin_image_features.detach(), vin_fake_text.detach())
        vin_real_target = torch.ones_like(vin_real_score)
        vin_fake_target = torch.zeros_like(vin_fake_score)
        vin_d_loss = vin_adversarial_loss(vin_real_score, vin_real_target) + vin_adversarial_loss(vin_fake_score, vin_fake_target)
        vin_discriminator_optimizer.zero_grad()
        vin_d_loss.backward()
        vin_discriminator_optimizer.step()

        # ---------- Train generator: caption cross-entropy + adversarial objective
        vin_generator.train()
        for vin_parameter in vin_discriminator.parameters():
            vin_parameter.requires_grad_(False)
        vin_fake_logits, vin_image_features = vin_generator(vin_images, vin_input_tokens)
        vin_fake_text = vin_fake_text_representation(vin_fake_logits, vin_generator.embedding, vin_mask_tokens)
        vin_fake_score = vin_discriminator(vin_image_features, vin_fake_text)
        vin_g_adv = vin_adversarial_loss(vin_fake_score, torch.ones_like(vin_fake_score))
        vin_g_caption = vin_caption_loss(vin_fake_logits.reshape(-1, vin_fake_logits.shape[-1]), vin_real_tokens.reshape(-1))
        vin_g_loss = vin_g_caption + 0.1 * vin_g_adv
        vin_generator_optimizer.zero_grad()
        vin_g_loss.backward()
        vin_generator_optimizer.step()
        for vin_parameter in vin_discriminator.parameters():
            vin_parameter.requires_grad_(True)
    print("Epoch", vin_epoch + 1, "D loss", round(vin_d_loss.item(), 4), "G loss", round(vin_g_loss.item(), 4))

torch.save({"generator": vin_generator.state_dict(), "vocabulary": vin_vocab.word_to_index}, "19_gan_image_caption.pth")
print("Saved generator. Gumbel-Softmax provides a differentiable approximation for discrete word choices.")
