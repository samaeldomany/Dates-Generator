import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConditionalVAE(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size, cond_emb_dim=16, date_emb_dim=32, hidden_dim=256, latent_dim=64, max_seq_len=12):
        super(ConditionalVAE, self).__init__()
        self.max_seq_len = max_seq_len
        self.date_vocab_size = date_vocab_size
        self.cond_embedding = nn.Embedding(cond_vocab_size, cond_emb_dim)
        self.date_embedding = nn.Embedding(date_vocab_size, date_emb_dim)
        self.encoder_lstm = nn.LSTM(date_emb_dim + (4 * cond_emb_dim), hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(date_emb_dim + (4 * cond_emb_dim) + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, date_vocab_size)

    def encode(self, x_cond, y_date):
        batch_size, seq_len = y_date.shape
        cond_emb = self.cond_embedding(x_cond)
        cond_emb_flat = cond_emb.view(batch_size, -1)
        cond_emb_seq = cond_emb_flat.unsqueeze(1).repeat(1, seq_len, 1)
        date_emb = self.date_embedding(y_date)
        enc_input = torch.cat([date_emb, cond_emb_seq], dim=-1)
        _, (h_n, _) = self.encoder_lstm(enc_input)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_cond, y_date_input):
        batch_size, seq_len = y_date_input.shape
        cond_emb = self.cond_embedding(x_cond).view(batch_size, -1)
        cond_emb_seq = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
        date_emb = self.date_embedding(y_date_input)
        dec_input = torch.cat([date_emb, cond_emb_seq, z_seq], dim=-1)
        lstm_out, _ = self.decoder_lstm(dec_input)
        out_logits = self.fc_out(lstm_out)
        return out_logits

    def forward(self, x_cond, y_date):
        mu, logvar = self.encode(x_cond, y_date)
        z = self.reparameterize(mu, logvar)
        decoder_input = y_date[:, :-1]
        out_logits = self.decode(z, x_cond, decoder_input)
        return out_logits, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    CE = F.cross_entropy(recon_x.transpose(1, 2), x, ignore_index=0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= x.size(0)
    beta = 0.1
    return CE + (beta * KLD)

def generate_dates(model, x_cond, preprocessor, device):
    model.eval()
    batch_size = x_cond.size(0)
    z = torch.randn(batch_size, model.fc_mu.out_features).to(device)
    sos_idx = preprocessor.date_vocab['<SOS>']
    current_sequence = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(preprocessor.max_date_len - 1):
            out_logits = model.decode(z, x_cond, current_sequence)
            next_char_logits = out_logits[:, -1, :]
            next_char_idx = torch.argmax(next_char_logits, dim=-1).unsqueeze(1)
            current_sequence = torch.cat([current_sequence, next_char_idx], dim=1)
    return current_sequence


class ConditionalGenerator(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size, cond_emb_dim=16, latent_dim=64, hidden_dim=256, seq_len=12):
        super(ConditionalGenerator, self).__init__()
        self.seq_len = seq_len
        self.date_vocab_size = date_vocab_size
        self.cond_embedding = nn.Embedding(cond_vocab_size, cond_emb_dim)
        in_features = latent_dim + (4 * cond_emb_dim)
        self.fc_init = nn.Linear(in_features, hidden_dim * seq_len)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, date_vocab_size)

    def forward(self, z, x_cond, tau=1.0):
        batch_size = z.size(0)
        cond_emb = self.cond_embedding(x_cond).view(batch_size, -1)
        gen_input = torch.cat([z, cond_emb], dim=1)
        seq_init = self.fc_init(gen_input).view(batch_size, self.seq_len, -1)
        lstm_out, _ = self.lstm(seq_init)
        logits = self.fc_out(lstm_out)
        gumbel_out = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        return gumbel_out

class ConditionalDiscriminator(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size, cond_emb_dim=16, hidden_dim=256):
        super(ConditionalDiscriminator, self).__init__()
        self.cond_embedding = nn.Embedding(cond_vocab_size, cond_emb_dim)
        in_features = date_vocab_size + (4 * cond_emb_dim)
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq_one_hot, x_cond):
        batch_size, seq_len, _ = seq_one_hot.shape
        cond_emb = self.cond_embedding(x_cond).view(batch_size, -1)
        cond_emb_seq = cond_emb.unsqueeze(1).repeat(1, seq_len, 1)
        disc_input = torch.cat([seq_one_hot, cond_emb_seq], dim=-1)
        _, (h_n, _) = self.lstm(disc_input)
        h_n_combined = torch.cat([h_n[0], h_n[1]], dim=1)
        out_logit = self.fc_out(h_n_combined)
        return out_logit

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class DateTransformer(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size, d_model=128, nhead=4, num_layers=2, max_seq_len=12):
        super(DateTransformer, self).__init__()
        self.d_model = d_model
        self.cond_embedding = nn.Embedding(cond_vocab_size, d_model)
        self.date_embedding = nn.Embedding(date_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 4)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, date_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_cond, y_date):
        batch_size, seq_len = y_date.shape
        cond_emb = self.cond_embedding(x_cond)
        date_emb = self.date_embedding(y_date) * math.sqrt(self.d_model)
        transformer_input = torch.cat([cond_emb, date_emb], dim=1)
        transformer_input = self.pos_encoder(transformer_input)
        total_len = transformer_input.size(1)
        src_mask = self.generate_square_subsequent_mask(total_len).to(y_date.device)
        out = self.transformer(transformer_input, mask=src_mask)
        date_out = out[:, 4:, :]
        logits = self.fc_out(date_out)
        return logits

class DateMLP(nn.Module):
    def __init__(self, cond_vocab_size, date_vocab_size, cond_emb_dim=16, hidden_dim=256, seq_len=12):
        super(DateMLP, self).__init__()
        self.seq_len = seq_len
        self.date_vocab_size = date_vocab_size
        self.cond_embedding = nn.Embedding(cond_vocab_size, cond_emb_dim)
        in_features = 4 * cond_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, seq_len * date_vocab_size)
        )

    def forward(self, x_cond):
        batch_size = x_cond.size(0)
        cond_emb = self.cond_embedding(x_cond).view(batch_size, -1)
        flat_output = self.net(cond_emb)
        seq_logits = flat_output.view(batch_size, self.seq_len, self.date_vocab_size)
        return seq_logits