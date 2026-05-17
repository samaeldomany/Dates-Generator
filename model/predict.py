import argparse
import os
import torch
from preprocessing import DateDataPreprocessor
from architectures import (DateTransformer, ConditionalVAE, ConditionalGenerator, 
                           DateMLP, generate_dates)

def generate_transformer_date(model, x_tensor, preprocessor, device):
    model.eval()
    batch_size = x_tensor.size(0)
    sos_idx = preprocessor.date_vocab['<SOS>']
    eos_idx = preprocessor.date_vocab['<EOS>']

    current_sequence = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(preprocessor.max_date_len - 1):
            logits = model(x_tensor, current_sequence)
            next_char_logits = logits[:, -1, :] 
            next_char_idx = torch.argmax(next_char_logits, dim=-1).unsqueeze(1)
            current_sequence = torch.cat([current_sequence, next_char_idx], dim=1)
            if next_char_idx.item() == eos_idx:
                break

    return current_sequence

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file path")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output file path")
    parser.add_argument('-m', '--model', type=str, choices=['transformer', 'vae', 'gan', 'mlp'], 
                        default='transformer', help="Which model to use for prediction")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = DateDataPreprocessor()
    data_path = os.path.join(os.path.dirname(__file__), '../Data/data.txt') 
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '../data/data.txt')
    preprocessor.fit(data_path)

    vocab_cond = len(preprocessor.cond_vocab)
    vocab_date = len(preprocessor.date_vocab)

    if args.model == 'transformer':
        model = DateTransformer(cond_vocab_size=vocab_cond, date_vocab_size=vocab_date).to(device)
        weights_name = 'transformer_weights.pt'
    elif args.model == 'vae':
        model = ConditionalVAE(cond_vocab_size=vocab_cond, date_vocab_size=vocab_date).to(device)
        weights_name = 'cvae_weights.pt'
    elif args.model == 'gan':
        model = ConditionalGenerator(cond_vocab_size=vocab_cond, date_vocab_size=vocab_date).to(device)
        weights_name = 'cgan_weights.pt'
    elif args.model == 'mlp':
        model = DateMLP(cond_vocab_size=vocab_cond, date_vocab_size=vocab_date).to(device)
        weights_name = 'mlp_weights.pt'

    weights_path = os.path.join(os.path.dirname(__file__), weights_name)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            conditions = [c.replace('[', '').replace(']', '').strip() for c in line.split() if c]
            x_encoded = [preprocessor.cond_vocab.get(c, 0) for c in conditions]
            x_tensor = torch.tensor([x_encoded], dtype=torch.long).to(device)

            with torch.no_grad():
                if args.model == 'transformer':
                    y_pred = generate_transformer_date(model, x_tensor, preprocessor, device)
                elif args.model == 'vae':
                    y_pred = generate_dates(model, x_tensor, preprocessor, device)
                elif args.model == 'gan':
                    z = torch.randn(x_tensor.size(0), 64).to(device)
                    logits = model(z, x_tensor)
                    y_pred = torch.argmax(logits, dim=-1)
                elif args.model == 'mlp':
                    logits = model(x_tensor)
                    y_pred = torch.argmax(logits, dim=-1)

            y_pred_list = y_pred[0].tolist()
            raw_date_str = preprocessor.decode_date(y_pred_list)

            parts = raw_date_str.split('-')
            if len(parts) == 3:
                y, m, d = parts[0], parts[1], parts[2]
                final_date_str = f"{d}-{m}-{y}"
            else:
                final_date_str = raw_date_str 

            outfile.write(f"{line} {final_date_str}\n")

if __name__ == "__main__":
    main()