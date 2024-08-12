# Inputs

The inputs folder contains essential files used for testing and analyzing the embeddings of the BART model. Each file in this directory serves a specific purpose in evaluating the performance and robustness of BART embeddings at different stages.
Directory Contents
### 1. bart_all_word_embedding.pkl

+ **Description:** This file contains the complete set of embeddings at **step 1** (*cf Bart README*) of the BART model for each word. The embeddings are stored as a map (Python dictionary), where each word is associated with its corresponding embedding.
+ **Format:** The file is in .pkl format, which means it is serialized using Python's pickle library. This format allows for easy storage and loading of complex objects, but be mindful of Python versions and dependencies to avoid compatibility issues when loading the file.

### 2. bart_vocab_with_ids.txt

+ **Description:** This file includes the full vocabulary of BART along with the associated IDs used by the tokenizer.
+ **Purpose:** This file is crucial for understanding how words are tokenized and mapped to their corresponding embeddings in BART.

### 3. english-common-words.txt

+ **Description:** A dictionary of 3,000 commonly used English words, used for testing random words.
+ **Limitation:** Although the dictionary consists of common words, some of these words might not appear in exactly this form in the BART tokenizer, which can sometimes make the tests inconclusive.

### 4. sentences_pair.json

+ **Description:** This file contains 50 pairs of sentences, structured as follows:
```json
    {
        "pair_id": 1,
        "sentence1": "A man is walking on the street.",
        "sentence2": "A woman is walking on the street."
    }
```

+ **Purpose:** Each pair contains two sentences that are identical except for one word, which does not drastically change the meaning of the sentence. This file is used to test how embeddings handle minor modifications in sentences.

### 5. word_pair.json

+ **Description:** This file contains 20 pairs of masculine/feminine words, structured as follows:
```json
{
    "word1": "king",
    "word2": "queen"
}
```
+ **Purpose:** This file is used to explore differences and similarities in the embeddings of words with gender-related relationships, such as "king" and "queen".
