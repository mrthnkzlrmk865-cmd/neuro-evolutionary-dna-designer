import numpy as np

class PromoterDatasetGenerator:
    """Generates synthetic DNA sequences and evaluates their biological expression scores."""
    def __init__(self, sequence_length=20, num_samples=1000):
        self.seq_len = sequence_length
        self.num_samples = num_samples
        self.bases = ['A', 'C', 'G', 'T']
        self.base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
    def generate_raw_sequences(self):
        return ["".join(np.random.choice(self.bases, self.seq_len)) for _ in range(self.num_samples)]
    
    def calculate_expression_score(self, sequence):
        score = 0.0
        if "TATA" in sequence:
            score += 4.0 + (10 / (sequence.find("TATA") + 1))
            
        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / self.seq_len
        if 0.4 <= gc_ratio <= 0.6:
            score += 3.0
        else:
            score -= 1.5
            
        if "GGGG" in sequence:
            score -= 2.0
            
        return 1 / (1 + np.exp(-score/2))

    def one_hot_encode(self, sequence):
        encoding = np.zeros((self.seq_len, 4))
        for i, base in enumerate(sequence):
            encoding[i, self.base_to_idx[base]] = 1.0
        return encoding.flatten()

    def create_dataset(self):
        sequences = self.generate_raw_sequences()
        X = np.array([self.one_hot_encode(seq) for seq in sequences])
        y = np.array([self.calculate_expression_score(seq) for seq in sequences]).reshape(-1, 1)
        return X, y, sequences
