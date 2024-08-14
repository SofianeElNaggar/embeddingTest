# Rebuild Embedding

This directory contains the code and scripts used for testing the embeddings of BART at **Step 1** in the process (*cf Bart README*). At this stage, the embeddings are tested before undergoing positional encoding and normalization, allowing for a more fundamental analysis of the embedding process.

## Files and Descriptions

### 1. `main.py`
- **Description**: This script is comparable to the one in the `full_embedding` directory. It contains several examples of how to use the functions defined in `input_embedding_tools.py`. It serves as the entry point for running various embedding tests and experiments at Step 1.
- **Usage**: You can run this script to execute predefined tests or modify it to explore different aspects of the BART embeddings at this early stage.

### 2. `input_embedding_tools.py`
- **Description**: This file contains several useful functions for working with the embeddings at **Step 1**. While not as extensive as the tools found in `tools.py` from the `full_embedding` directory, these functions are tailored for analyzing the raw embeddings before they are modified by positional encoding and normalization.
- **Functionality**: The functions here allow for the extraction and basic manipulation of the initial embeddings, providing a foundational understanding of how BART's embeddings behave prior to any transformations.

## Overview

The tests in this directory represent a more fundamental approach to understanding BART's embeddings, focusing on their state before any modifications are applied. This allows for a potentially more accurate and insightful analysis of the embedding process.
