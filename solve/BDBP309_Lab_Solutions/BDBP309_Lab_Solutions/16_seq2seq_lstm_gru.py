"""Lab 16: Sequence-to-sequence translation with LSTM and GRU.

Optional translation_pairs.csv columns: source,target (e.g. English,Hindi transliteration).
Without the CSV, an illustrative tiny phrase-pair dataset is used.
"""
import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_pairs = []
if os.path.exists("translation_pairs.csv"):
    vin_df = pd.read_csv("translation_pairs.csv")
    for vin_i in range(len(vin_df)):
        vin_pairs.append((str(vin_df.iloc[vin_i]["source"]), str(vin_df.iloc[vin_i]["target"])))
else:
    vin_pairs = [
        ("hello", "namaste"), ("good morning", "suprabhat"),
        ("thank you", "dhanyavaad"), ("how are you", "kaise ho"),
        ("good night", "shubh ratri"), ("i am fine", "main thik hoon")
    ]
    print("Using a tiny demonstration corpus; add translation_pairs.csv for real training.")

class VinVocabulary:
    def __init__(self, vin_texts):
        self.word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        for vin_text in vin_texts:
            for vin_word in re.findall(r"[\w]+", vin_text.lower()):
                if vin_word not in self.word_to_index:
                    self.word_to_index[vin_word] = len(self.word_to_index)
    def encode(self, vin_text, vin_max_length):
        vin_result = [1]
        for vin_word in re.findall(r"[\w]+", vin_text.lower()):
            vin_result.append(self.word_to_index.get(vin_word, 3))
        vin_result.append(2)
        vin_result = vin_result[:vin_max_length]
        while len(vin_result) < vin_max_length:
            vin_result.append(0)
        return torch.tensor(vin_result, dtype=torch.long)

vin_source_vocab = VinVocabulary([vin_pair[0] for vin_pair in vin_pairs])
vin_target_vocab = VinVocabulary([vin_pair[1] for vin_pair in vin_pairs])
vin_max_length = 12

class VinTranslationDataset(Dataset):
    def __len__(self):
        return len(vin_pairs)
    def __getitem__(self, vin_index):
        vin_source, vin_target = vin_pairs[vin_index]
        return vin_source_vocab.encode(vin_source, vin_max_length), vin_target_vocab.encode(vin_target, vin_max_length)

vin_loader = DataLoader(VinTranslationDataset(), batch_size=4, shuffle=True)

class VinSeq2Seq(nn.Module):
    def __init__(self, vin_cell_type, vin_source_size, vin_target_size):
        super().__init__()
        self.cell_type = vin_cell_type
        self.source_embedding = nn.Embedding(vin_source_size, 64, padding_idx=0)
        self.target_embedding = nn.Embedding(vin_target_size, 64, padding_idx=0)
        if vin_cell_type == "LSTM":
            self.encoder = nn.LSTM(64, 128, batch_first=True)
            self.decoder = nn.LSTM(64, 128, batch_first=True)
        else:
            self.encoder = nn.GRU(64, 128, batch_first=True)
            self.decoder = nn.GRU(64, 128, batch_first=True)
        self.output = nn.Linear(128, vin_target_size)
    def forward(self, vin_source, vin_target_input):
        vin_source_embedded = self.source_embedding(vin_source)
        vin_encoder_output, vin_state = self.encoder(vin_source_embedded)
        vin_target_embedded = self.target_embedding(vin_target_input)
        vin_decoder_output, vin_state = self.decoder(vin_target_embedded, vin_state)
        return self.output(vin_decoder_output)

vin_final_losses = {}
for vin_cell_type in ["LSTM", "GRU"]:
    vin_model = VinSeq2Seq(vin_cell_type, len(vin_source_vocab.word_to_index), len(vin_target_vocab.word_to_index)).to(vin_device)
    vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.003)
    vin_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    for vin_epoch in range(100):
        vin_model.train()
        vin_total_loss = 0.0
        for vin_source, vin_target in vin_loader:
            vin_source = vin_source.to(vin_device)
            vin_target = vin_target.to(vin_device)
            vin_logits = vin_model(vin_source, vin_target[:, :-1])
            vin_expected = vin_target[:, 1:]
            vin_loss = vin_loss_function(vin_logits.reshape(-1, vin_logits.shape[-1]), vin_expected.reshape(-1))
            vin_optimizer.zero_grad()
            vin_loss.backward()
            vin_optimizer.step()
            vin_total_loss = vin_total_loss + vin_loss.item()
        if vin_epoch % 20 == 0:
            print(vin_cell_type, "epoch", vin_epoch, "loss", vin_total_loss / len(vin_loader))
    vin_final_losses[vin_cell_type] = vin_total_loss / len(vin_loader)
    torch.save(vin_model.state_dict(), "16_seq2seq_" + vin_cell_type.lower() + ".pth")

print("Final training loss comparison:", vin_final_losses)
