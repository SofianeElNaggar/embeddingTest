# Results Directory

This directory contains the results of various experiments conducted to analyze the behavior and characteristics of BART embeddings. All these tests were performed at step 2 of the embedding process, meaning they include positional encoding and normalization. As a result, the findings are generally not highly relevant for understanding the raw embedding structure but provide insight into the post-processed embeddings.

### 1. `change_dim`

This directory contains JSON files documenting the results of experiments where we altered only one dimension of the embedding vector at a time. For each dimension in the vector, we modified it while keeping all other dimensions constant to observe the impact. The procedure works as follows:

- If an embedding vector is `[1, 4, 2]`, we generate variations such as `[0, 4, 2]`, `[1, 4, 2]`, ..., `[n, 4, 2]`, `[1, 0, 2]`,... ,`[1, n, 2]`and so on.
- For each modified vector, we decode it back to the closest word(s) and record the list of words obtained.

The goal of this experiment was to determine whether certain dimensions in the embedding vector carry specific, consistent meanings (e.g., emotion, gender, etc.).

### 2. `distances`

This directory contains graphs and JSON files that record the distance statistics derived from the `sentences_pair.json` file. The analyses include:

- Distances between the words that differ between the sentence pairs.
- Distances between the surrounding words of the differing words.
- Random word distances.

Additionally, it contains information on which dimensions change the most between words.

### 3. `interpolation`

This directory contains JSON files documenting the results of interpolation experiments between two words in the embedding space. The aim was to analyze the transition of meanings as one word's embedding gradually shifts towards another.

### 4. `neighbor`

This directory contains JSON files that test the nearest neighbors of a word in the embedding space at different distances. This helps to understand the semantic neighborhood of a word within the embedding space.

## Notes

- **Step 2 Testing**: All experiments were conducted at step 2, meaning that the embeddings were already processed with positional encoding and normalization. This post-processing may obscure the raw embedding structure and introduce biases, making the results less relevant for interpreting the core embedding properties.
