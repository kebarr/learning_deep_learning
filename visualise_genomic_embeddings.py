To visualize your genomic embeddings using UMAP or t-SNE, we need to pull the weights matrix from your model's embedding layer (where each row represents a unique k-mer's vector) and compress those hundreds of hidden dimensions down to 2D space.

Here is a complete, self-contained pipeline to extract the embeddings, reduce them using UMAP or t-SNE, and plot them interactively so you can see which k-mers cluster together.
Step 1: Install the Required Libraries

If you don't have UMAP installed, you can grab it via pip (it's faster and often preserves global structure better than t-SNE for biological data):
Bash

pip install umap-learn plotly scikit-learn

Step 2: The Visualization Script

Run this script after your Learner has been created and trained (even slightly).
Python

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import umap

def visualize_genomic_embeddings(learn, method='umap'):
    """
    Extracts k-mer embeddings from a fastai learner and plots them in 2D.
    method: 'umap' or 'tsne'
    """
    # 1. Extract the embedding weights from the fastai model
    # For standard fastai text learners, the encoder is the first element of the architecture
    encoder = learn.model[0].encoder
    weights = encoder.weight.detach().cpu().numpy()
    
    # 2. Get the corresponding k-mer tokens from the vocabulary
    # vocab[0] holds the token strings mapped to those embedding rows
    kmers = list(learn.dls.vocab[0])
    
    # Ensure our weights matrix matches our vocabulary size 
    # (fastai sometimes appends padding tokens to the end)
    weights = weights[:len(kmers)]
    
    print(f"Extracted embedding matrix of shape: {weights.shape}")
    print(f"Reducing dimensions using {method.upper()}...")
    
    # 3. Apply Dimensionality Reduction
    if method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(weights)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embedding_2d = reducer.fit_transform(weights)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'")
        
    # 4. Create a DataFrame for Plotly
    df_plot = pd.DataFrame({
        'X': embedding_2d[:, 0],
        'Y': embedding_2d[:, 1],
        'kmer': kmers
    })
    
    # Optional: Highlight special fastai tokens so they don't confuse you
    df_plot['Type'] = df_plot['kmer'].apply(
        lambda x: 'Special Token' if x.startswith('xx') else 'Genomic k-mer'
    )
    
    # 5. Plot interactively using Plotly
    fig = px.scatter(
        df_plot, 
        x='X', 
        y='Y', 
        text='kmer', 
        color='Type',
        title=f"2D Projection of Genomic k-mer Embeddings ({method.upper()})",
        labels={'X': 'Component 1', 'Y': 'Component 2'},
        hover_data=['kmer']
    )
    
    # Adjust text position and marker appearance to make it readable
    fig.update_traces(textposition='top center', marker=dict(size=8, opacity=0.7))
    fig.update_layout(template='plotly_white', width=1000, height=800)
    
    fig.show()
    
    return df_plot

Step 3: Running it on your Learner

Once you've run your training loops (learn.fine_tune or learn.fit_one_cycle), you simply pass your learn object directly to the function:
Python

# To visualize using UMAP
df_embeddings = visualize_genomic_embeddings(learn, method='umap')

# Or if you want to contrast it with t-SNE
# df_embeddings = visualize_genomic_embeddings(learn, method='tsne')

What to look for in the resulting plot:

Because it is an interactive Plotly graph, you can zoom in and hover over the dots. In a well-trained genomic model, you will typically observe biological clustering:

    GC-Content Splits: You'll often see a distinct gradient where AT-rich k-mers cluster on one side of the space and GC-rich k-mers cluster on the other.

    Synonymous Codons: If you are using k=3, codons that translate to the same amino acid (like GCT, GCC, GCA, and GCG all coding for Alanine) will frequently gravitate toward the exact same neighborhood because they share similar contextual "meaning" in the coding sequences.