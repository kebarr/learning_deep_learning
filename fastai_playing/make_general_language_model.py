
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from fastai.text.all import *

# ==============================================================================
# 1. CLEAN AND PREPARE DATA
# ==============================================================================
df = pd.read_csv('/home/katie/learning_deep_learning/miclassified_reads_all/shigellae/all_species_with_qual.csv', sep="\t")
df = df.dropna(subset=['sequence', 'quality'])

# Reset index to avoid any missing row index errors
df = df.reset_index(drop=True)

# ==============================================================================
# 2. RUN OUR QUALITY K-MER TOKENIZER IN VANILLA PYTHON
# ==============================================================================
print("Tokenizing genomic reads with quality scores...")
k = 3
low_thresh = 20
high_thresh = 30

all_tokenized_reads = []

for idx, row in df.iterrows():
    seq = row['sequence']
    qual = row['quality']
    
    if len(seq) < k or len(seq) != len(qual):
        all_tokenized_reads.append(['xxpad']) # Safeguard fallback
        continue
        
    tuple_kmers = []
    for i in range(len(seq) - k + 1):
        kmer_seq = seq[i:i+k]
        kmer_qual = qual[i:i+k]
        
        # ASCII Phred scores conversion
        phred_scores = [ord(char) - 33 for char in kmer_qual]
        avg_backbone_qual = np.mean(phred_scores)
        
        if avg_backbone_qual >= high_thresh:
            qual_tag = "H"
        elif avg_backbone_qual <= low_thresh:
            qual_tag = "L"
        else:
            qual_tag = "M"
            
        tuple_kmers.append(f"{kmer_seq}_{qual_tag}")
    
    all_tokenized_reads.append(tuple_kmers)

# ==============================================================================
# 3. MANUALLY BUILD VOCABULARY MAPS
# ==============================================================================
# Flatten all tokens to find unique elements and build vocabulary
flat_tokens = [tok for read in all_tokenized_reads for tok in read]
unique_tokens = sorted(list(set(flat_tokens)))

# Add fastai's core special tokens at the front
vocab_x = ['xxpad', 'xxunk'] + unique_tokens
tok_to_id = {tok: idx for idx, tok in enumerate(vocab_x)}

# Build Label Vocabulary
unique_labels = sorted(list(df['label'].unique()))
vocab_y = unique_labels
label_to_id = {lbl: idx for idx, lbl in enumerate(vocab_y)}

# ==============================================================================
# 4. NUMERICALIZE AND PAD SEQUENCES
# ==============================================================================
print("Converting tokens into GPU integer IDs...")
numerical_x = []
for read in all_tokenized_reads:
    numerical_x.append([tok_to_id.get(tok, 1) for tok in read]) # 1 is 'xxunk'

# Pad everything to match the length of the longest read sequence in your batch
max_len = max(len(read) for read in numerical_x)
padded_x = [read + [0] * (max_len - len(read)) for read in numerical_x] # 0 is 'xxpad'

# Convert everything to standard PyTorch LongTensors
X_tensor = torch.tensor(padded_x, dtype=torch.long)
Y_tensor = torch.tensor([label_to_id[lbl] for lbl in df['label']], dtype=torch.long)

# ==============================================================================
# 5. ASSEMBLE RAW FASTAI DATALOADERS
# ==============================================================================
print("Assembling DataLoaders...")

# Use a standard random split index array
set_seed(42)
splits = RandomSplitter(valid_pct=0.2)(df)
train_idx, valid_idx = splits[0], splits[1]

# Slice tensors into explicit training and validation datasets
train_ds = TensorDataset(X_tensor[train_idx], Y_tensor[train_idx])
valid_ds = TensorDataset(X_tensor[valid_idx], Y_tensor[valid_idx])

# Build fastai standard DataLoaders directly from the datasets
dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=64, shuffle=True)

# Inject the vocabularies manually so fastai interpretation tools still function
dls.vocab = [vocab_x, vocab_y]