# Here is a custom Transform class designed to handle k-mers.

# It splits a continuous DNA/RNA string into overlapping substrings of length k. For example, if you choose k=3 (codons), the sequence ATCGAT will be tokenized into ['ATC', 'TCG', 'CGA', 'GAT'].

# Because you are using k-mers, your vocabulary will now consist of short words rather than single letters. Instead of hardcoding the vocabulary, we can let fastai automatically build it from your data, while ensuring our special padding tokens are preserved.
# Python

from fastai.text.all import *

# 1. Custom Transform for Overlapping k-mers
class GenomicKmerTokenizer(Transform):
    def __init__(self, k=3):
        self.k = k
        
    def encodes(self, x):
        # Gracefully handle empty or short sequences
        if not isinstance(x, str) or len(x) < self.k:
            return L()
        
        # Extract overlapping k-mers using a list comprehension
        kmers = [x[i:i+self.k] for i in range(len(x) - self.k + 1)]
        return L(kmers)

# 2. Flexible Block Constructor
def GenomicKmerBlock(k=3):
    return TransformBlock(
        type_tfms=GenomicKmerTokenizer(k=k),
        # Passing min_freq=1 ensures all discovered k-mers are included in the vocabulary
        item_tfms=Numericalize(min_freq=1) 
    )

# How to use it in your flexible pipeline

# When you build your DataBlock now, you don't need to specify a hardcoded vocabulary array. fastai will automatically read your CSV, break the strings into your specified k-mer size, and build the dictionary mapping for you.
# Python

# Setup your block (e.g., using 3-mers)
genomics_db = DataBlock(
    blocks=(GenomicKmerBlock(k=3), CategoryBlock),  
    get_x=ColReader('sequence'),               
    get_y=ColReader('label'),                  
    splitter=RandomSplitter(valid_pct=0.2, seed=42)
)

# Load your prepared data
df = pd.read_csv('all_misclassified_Bacillus_anthracis_prepared.csv')
dls = genomics_db.dataloaders(df, bs=64, before_batch=pad_input_chunk)

# Inspect your new k-mer vocabulary
print("Total unique k-mers found:", len(dls.vocab[0]))
print("Sample k-mers:", dls.vocab[0][:15])

# Why this approach is ideal:

#     Tunable k: You can easily change GenomicKmerBlock(k=3) to k=4 or k=6 depending on how granular you want your sequence context windows to be.

#     Auto-vocab calculation: Manually calculating every possible permutation of a 4-mer or 5-mer vocabulary is tedious (44=256 combinations, 46=4096). Setting Numericalize(min_freq=1) lets fastai do the heavy lifting safely during setup.