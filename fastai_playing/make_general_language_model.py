
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from fastai.text.all import *

# going back to the The Universal Language Model - should I label the species or just combine all reads completely unlabelled

# For the Universal Language Model (Pre-training) stage, you should combine all the reads completely unlabeled, treating them as one giant soup of genomic text.

# You do not want to label the species or the true/false status here.

# Why Completely Unlabeled?
# The entire goal of a Language Model (LM) is self-supervised learning. The model is playing a game of "predict the next k-mer." By removing all labels and mixing the species together, you force the neural network to look strictly at the syntax, grammar, and patterns of DNA sequences themselves.

# If you were to inject species labels at this stage, the model would try to shortcut its learning by associating specific error types with those macro labels, instead of learning the universal "rules" of sequencer data and genomic structure.

# The Workflow Visualized
# Think of the pre-training strategy as a two-step assembly line:

# Plaintext
# STEP 1: Pre-training (The Generalist)
# [Species A Reads] \
# [Species B Reads]  -->  [ Combined Unlabeled Data ]  -->  Train Language Model
# [Species C Reads] /                                        (Learns DNA Grammar)
#                                                                     |
#                                                                     v
# STEP 2: Fine-Tuning (The Specialist)                       [ Extract Backbone ]
#                                                                     |
# [Species A True/False Reads Only] ----------------------------------+--> Train Gatekeeper Filter
#                                                                          (Surgically models Species A errors)
# In Step 1 (Language Model): The model reads strings like ATG_H CCG_M TTA_L and learns that after a certain motif, a specific low-quality k-mer is highly probable due to common machine physics. It becomes an expert in "the language of sequencing."

# In Step 2 (Classifier Fine-Tuning): You load those pre-trained weights. Now you introduce your species-specific True/False labeled dataset. Because the model already understands what normal sequencing reads look like, it can immediately focus its attention on the exact anomalies causing your false positives.

# When you start prepping that combined dataset, just extract the sequence and quality strings from a random, equal subset of reads from each species, concatenate them with the | delimiter we built, and feed them into a text language learner model block!

# should I just include species I want to find, or their close relatives as well?

# For the Universal Language Model (Pre-training), you should absolutely include both the species you want to find AND their close relatives. In fact, including close relatives—and even known background contaminants—is one of the best ways to make your final gatekeeper highly accurate.

# Here is the exact biological and machine learning reasoning for why including relatives makes your language model vastly superior.

# 1. It Sharpens the Model's "Vocabulary Boundaries"
# If you only train the language model on the target species (e.g., Bacillus anthracis), the model learns a very narrow vocabulary. When it later encounters a false positive read from a close relative (like Bacillus cereus), the model might look at it and say, "Well, this looks 95% like my target, so it must be normal target DNA." By including Bacillus cereus and Bacillus thuringiensis in the unlabeled pre-training soup, the language model learns the subtle, hyper-specific k-mer shifts that differentiate these species. It learns the "shared grammar" of the Bacillus genus while mapping the exact boundary variations between them.

# 2. It Directly Models "Homology Traps"
# Many species-level false positives happen because of homology—highly conserved genes (like the 16S rRNA gene, housekeeping genes, or shared plasmids) that are practically identical across related species.

# If close relatives are in your pre-training data, the language model learns that these specific conserved sequences are universal and highly predictable across multiple genomic contexts. When you later fine-tune your classifier, the model will know not to place too much diagnostic weight on those generic, shared regions, forcing the classifier to look for the truly unique, species-defining k-mers instead.

# 3. It Better Mimics Your Lab Environment
# In a real lab sequencing run, cross-contamination, sample bleed, and index hopping happen most frequently between samples processed in the same batch—which are often related projects or organisms.

# By pooling your target species, their close relatives, and even common lab background "weeds" (like E. coli or Human DNA if you work with clinical/environmental samples) into the unlabeled pre-training mix, you teach the model to expect the exact background noise it will have to filter out in production.

# The Recommended Pre-Training "Soup" Recipe
# When you create your big unlabeled dataframe for Step 1, aim for a balanced mix that looks like this:

# 40% Target Species: A broad sampling of your primary organisms of interest.

# 40% Close Relatives: Sister species, or species within the same genus/family that often cause misclassifications.

# 20% Common Background Contaminants: Universal sequencing background data (e.g., host DNA, common vector sequences, extraction kit flora).

# By making the generalist model study this diverse pool, you build an incredibly robust genomic backbone. When you finally fine-tune it on your target species' tiny <100 False Positive dataset, the model will instantly recognize exactly which reads are genuine targets and which ones are sneaky, high-homology relatives slipping through the gate!

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

lm_block = DataBlock(
    blocks=TextBlock(tok_tfm=GenomicKmerTokenizer(k=3), is_lm=True),
    get_x=ColReader('sequence'),
    splitter=RandomSplitter(valid_pct=0.1)
)
lm_dls = lm_block.dataloaders(df, path='.', bs=64)

# Create a Language Model Learner
learn = language_model_learner(lm_dls, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()])
learn.fit_one_cycle(3, 2e-2) # this takes ages. probably cos not suitable

import pickle
from pathlib import Path

# Create a directory for your checkpoints if it doesn't exist
save_dir = Path("/home/katie/learning_deep_learning/fastai_playing/models")
save_dir.mkdir(exist_ok=True)

# 1. Save the PyTorch Model Weights
# fastai's native learn.save looks inside your path directory
learn.path = save_dir.parent
learn.save('genomic_gatekeeper_v1')
print("[+] Model architecture weights saved safely.")

# 2. Save your custom quality-aware Vocabs (Crucial!)
# We pack vocab_x and vocab_y together into a pickle binary file
vocab_package = {
    'vocab_x': vocab_x,
    'vocab_y': vocab_y
}

with open(save_dir / 'genomic_metadata.pkl', 'wb') as f:
    pickle.dump(vocab_package, f)
    
print("[+] Custom vocabulary dictionaries saved safely.")
print(">>> Safe to turn off your computer! See you tomorrow.")