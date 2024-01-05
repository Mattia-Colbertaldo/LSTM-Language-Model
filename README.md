# # Language Models with LSTM
Language model from scratch using Long Short-Term Memory (LSTM) networks with Pytorch

* ***Please take a look at the pdf for an insightful explaination with figures.***

The project is divided into several sections:

## Data Processing

1. **Data Download and Filtering (Line 21):** Downloaded and filtered news data labeled as "POLITICS." After inspection, 35,602 sequences were obtained.

2. **Tokenization (Line 52):** Tokenized headlines at a word level, converted to lowercase, and appended `<EOS>` at the end of each sentence. Tokenized sentences were saved in pickle format.

3. **Vocabulary Creation (Line 112):** Created a vocabulary containing unique words, assigned indices, and saved word-to-index and index-to-word dictionaries in pickle format.

4. **Dataset and DataLoader (Line 214, Line 248):** Implemented a dataset class and a collate function for padding sequences. Utilized PyTorch DataLoader for efficient batch processing.

## Model Definition

5. **LSTM Model (Line 270):** Designed a language model with an Embedding layer, LSTM layer, Dropout layer for regularization, and a fully connected (Linear) layer. Implemented `init_state` method for LSTM state initialization.

## Evaluation - Part 1

6. **Sampling Functions (Line 316):** Implemented functions for sampling strategies: `random_sample_next` for top-k sampling and `sample_argmax` for argmax sampling. Created a main `sample` function to generate sentences.

## Training

7. **Model Parameters and Training (Line 381):** Trained the model for 12 epochs with specified parameters: hidden_size=1024, emb_dim=150, n_layers=2, dropout_p=0.2. Used Cross-Entropy Loss, learning rate 0.001, and gradient clipping.

8. **TBTT Training (Line 430):** Conducted training using TBTT (Truncated Backpropagation Through Time) with hidden_size=2048, emb_dim=150, n_layers=1, dropout_p=0. Trained for 7 epochs with similar parameters as the regular training.

## Evaluation - Part 2

9. **Sentence Generation (Line 531):** Generated sentences for different prompts using both argmax and top-k sampling strategies.

## Bonus Question

10. **Vector Operations (Line 593):** Attempted to reproduce the mathematical operation "vector('King') - vector('Man') + vector('Woman')" and reported results, including the L2 distance and closest words.

## Questions

11. **Byte-Pair Encoding (BPE) and WordPiece (Line 621):** Explained the techniques of BPE and WordPiece, highlighting their role in breaking words into smaller parts during language processing and their advantages over simple word tokenization.

Feel free to explore the code and results for a detailed understanding of the language model and its performance.
