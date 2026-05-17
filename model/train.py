import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim as optim
import torch.nn.functional as F

from preprocessing import DateDataPreprocessor, DateDataset, ConditionValidator
from architectures import (ConditionalVAE, vae_loss_function, ConditionalGenerator,
                           ConditionalDiscriminator, DateTransformer, DateMLP, generate_dates)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/data.txt')
MODEL_DIR = os.path.join(BASE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessor = DateDataPreprocessor()
preprocessor.fit(DATA_PATH)
dataset = DateDataset(DATA_PATH, preprocessor, reorder_date=True)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

EPOCHS = 50
LEARNING_RATE = 0.001

model = ConditionalVAE(
    cond_vocab_size=len(preprocessor.cond_vocab),
    date_vocab_size=len(preprocessor.date_vocab)
).to(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out_logits, mu, logvar = model(x_batch, y_batch)
        y_target = y_batch[:, 1:]
        loss = vae_loss_function(out_logits, y_target, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_x, _ = next(iter(dataloader))
    val_x = val_x.to(device)
    generated_sequences = generate_dates(model, val_x, preprocessor, device)
    passed_count = 0
    total_count = val_x.size(0)

    for i in range(total_count):
        cond_text = [preprocessor.cond_inverse[idx.item()] for idx in val_x[i]]
        date_text = preprocessor.decode_date(generated_sequences[i].tolist())
        is_valid, _ = ConditionValidator.validate_date(date_text, cond_text, is_reordered=True)
        if is_valid:
            passed_count += 1

torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'cvae_weights.pt'))

GAN_EPOCHS = 50
LR_G = 0.0005
LR_D = 0.0001

generator = ConditionalGenerator(len(preprocessor.cond_vocab), len(preprocessor.date_vocab)).to(device)
discriminator = ConditionalDiscriminator(len(preprocessor.cond_vocab), len(preprocessor.date_vocab)).to(device)

opt_G = optim.Adam(generator.parameters(), lr=LR_G)
opt_D = optim.Adam(discriminator.parameters(), lr=LR_D)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(GAN_EPOCHS):
    generator.train()
    discriminator.train()
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        current_batch_size = x_batch.size(0)
        
        real_labels = torch.full((current_batch_size, 1), 0.9).to(device)
        fake_labels = torch.full((current_batch_size, 1), 0.1).to(device)

        opt_D.zero_grad()
        real_one_hot = F.one_hot(y_batch, num_classes=len(preprocessor.date_vocab)).float()
        d_real_out = discriminator(real_one_hot, x_batch)
        d_real_loss = criterion(d_real_out, real_labels)

        z = torch.randn(current_batch_size, 64).to(device)
        fake_one_hot = generator(z, x_batch)
        d_fake_out = discriminator(fake_one_hot.detach(), x_batch)
        d_fake_loss = criterion(d_fake_out, fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        if batch_idx % 2 == 0:
            opt_D.step()

        opt_G.zero_grad()
        g_fake_out = discriminator(fake_one_hot, x_batch)
        g_loss = criterion(g_fake_out, real_labels)
        g_loss.backward()
        opt_G.step()

torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'cgan_weights.pt'))

TRANSFORMER_EPOCHS = 30
LR_T = 0.001

transformer_model = DateTransformer(
    cond_vocab_size=len(preprocessor.cond_vocab),
    date_vocab_size=len(preprocessor.date_vocab)
).to(device)

optimizer_T = torch.optim.Adam(transformer_model.parameters(), lr=LR_T)
criterion_T = nn.CrossEntropyLoss(ignore_index=preprocessor.date_vocab['<PAD>'])

for epoch in range(TRANSFORMER_EPOCHS):
    transformer_model.train()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer_T.zero_grad()
        decoder_input = y_batch[:, :-1]
        targets = y_batch[:, 1:]
        logits = transformer_model(x_batch, decoder_input)
        loss = criterion_T(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer_T.step()

torch.save(transformer_model.state_dict(), os.path.join(MODEL_DIR, 'transformer_weights.pt'))

MLP_EPOCHS = 30
LR_MLP = 0.001

mlp_model = DateMLP(
    cond_vocab_size=len(preprocessor.cond_vocab),
    date_vocab_size=len(preprocessor.date_vocab)
).to(device)

optimizer_M = torch.optim.Adam(mlp_model.parameters(), lr=LR_MLP)
criterion_M = nn.CrossEntropyLoss(ignore_index=preprocessor.date_vocab['<PAD>'])

for epoch in range(MLP_EPOCHS):
    mlp_model.train()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer_M.zero_grad()
        logits = mlp_model(x_batch)
        loss = criterion_M(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
        loss.backward()
        optimizer_M.step()

torch.save(mlp_model.state_dict(), os.path.join(MODEL_DIR, 'mlp_weights.pt'))