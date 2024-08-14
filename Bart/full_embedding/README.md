# Full Embedding

This directory contains the code and scripts used for testing the embeddings of BART at **Step 2** in the process (*cf Bart README*). At this stage, the embeddings have been processed with positional encoding and normalization.


### 1. main.py
- **Description**: This is the main script, containing several examples of how to use the functions defined in `tools.py`. It serves as the entry point for running various embedding tests and experiments.
- **Usage**: You can run this script to execute predefined tests or modify it to explore different aspects of the BART embeddings.

### 2. tools.py
- **Description**: This file contains all the functions used to obtain BART embeddings and test them in various ways. It includes methods for extracting embeddings, applying different tests, and analyzing the results.
- **Functionality**: The functions here are designed to work with the embeddings generated at Step 2, allowing for detailed analysis and experimentation.

The tests in this directory represent our initial approach to understanding BART's embeddings after they have undergone positional encoding and normalization. While these tests are relatively naive, they provided a solid foundation for further exploration and deeper understanding of how BART embeddings work in practice.
