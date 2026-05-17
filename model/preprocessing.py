import re
import torch
import datetime
from torch.utils.data import Dataset

class DateDataPreprocessor:
    def __init__(self):
        self.cond_vocab = {'<PAD>': 0}
        self.cond_inverse = {0: '<PAD>'}
        self.date_chars = ['<PAD>', '<SOS>', '<EOS>', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.date_vocab = {char: idx for idx, char in enumerate(self.date_chars)}
        self.date_inverse = {idx: char for char, idx in self.date_vocab.items()}
        self.max_date_len = 12

    def fit(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            conditions = re.findall(r'\[(.*?)\]', line)
            for cond in conditions:
                if cond not in self.cond_vocab:
                    idx = len(self.cond_vocab)
                    self.cond_vocab[cond] = idx
                    self.cond_inverse[idx] = cond

    def transform_line(self, line, reorder_date=False):
        line = line.strip()
        if not line: return None, None
        parts = line.split(']')
        conditions = [p.replace('[', '').strip() for p in parts[:-1]]
        date_str = parts[-1].strip()
        x_encoded = [self.cond_vocab[c] for c in conditions]
        if reorder_date and '-' in date_str:
            d, m, y = date_str.split('-')
            date_str = f"{y}-{m}-{d}"
        y_encoded = [self.date_vocab['<SOS>']]
        y_encoded.extend([self.date_vocab[char] for char in date_str])
        y_encoded.append(self.date_vocab['<EOS>'])
        while len(y_encoded) < self.max_date_len:
            y_encoded.append(self.date_vocab['<PAD>'])
        return x_encoded, y_encoded

    def decode_date(self, y_encoded):
        chars = []
        for idx in y_encoded:
            char = self.date_inverse[idx]
            if char == '<EOS>': break
            if char not in ['<SOS>', '<PAD>']:
                chars.append(char)
        return "".join(chars)

class DateDataset(Dataset):
    def __init__(self, file_path, preprocessor, reorder_date=True):
        self.preprocessor = preprocessor
        self.reorder_date = reorder_date
        with open(file_path, 'r') as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        x_encoded, y_encoded = self.preprocessor.transform_line(line, self.reorder_date)
        x_tensor = torch.tensor(x_encoded, dtype=torch.long)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        return x_tensor, y_tensor

class ConditionValidator:
    @staticmethod
    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @staticmethod
    def validate_date(generated_date_str, conditions, is_reordered=True):
        if is_reordered:
            try:
                y, m, d = generated_date_str.split('-')
                standard_date_str = f"{d}-{m}-{y}"
            except ValueError:
                return False, "Format Error"
        else:
            standard_date_str = generated_date_str

        try:
            date_obj = datetime.datetime.strptime(standard_date_str, "%d-%m-%Y")
        except ValueError:
            return False, "Invalid Calendar Date"

        day_cond = conditions[0]
        month_cond = conditions[1]
        leap_cond = conditions[2]
        decade_cond = conditions[3]

        actual_day = date_obj.strftime('%a').upper()
        if actual_day != day_cond.upper():
            return False, "Day Mismatch"

        actual_month = date_obj.strftime('%b').upper()
        if actual_month != month_cond.upper():
            return False, "Month Mismatch"

        actual_leap = str(ConditionValidator.is_leap_year(date_obj.year))
        if actual_leap.lower() != leap_cond.lower():
            return False, "Leap Mismatch"

        if not str(date_obj.year).startswith(decade_cond):
            return False, "Decade Mismatch"

        return True, "Passed"