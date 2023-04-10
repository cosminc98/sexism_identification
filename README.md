# Text Classification in Romanian Language

This repository contains the code for our team's submission to a Natural Language Processing (NLP) competition ([Nitro](https://www.kaggle.com/competitions/nitro-language-processing-2)) hosted on Kaggle . The competition challenged participants to develop a pipeline for sexism text identification in the Romanian language.

## Competition Details
The task in this competition was to classify each text into one of `five` possible categories: (0) `Sexist Direct`, (1) `Sexist Descriptive`, (2) `Sexist Reporting`, (3) `Non-sexist Offensive`, and (4) `Non-sexist Non-offensive`.

The data for this competition has been collected from a variety of sources, including social media networks such as Facebook, Twitter, and Reddit, web articles, and books.

## Training Data
The training dataset provided for this competition consists of `40,000` text files from [CoRoSeOf](https://aclanthology.org/2022.lrec-1.243): *An annotated Corpus of Romanian Sexist and Offensive Language*, while the test set comprises `3130` text files.
  
Participants were expected to use the training data to build a pipeline that can accurately classify the text documents in the test set into the appropriate category.

The submission was evaluated based on `weighted accuracy`, with the tiebreaker based on the count of false negatives in identifying offensive language.

## Our Approach
Our team's `approach` consisted of the following steps:

- `Data sanitization`: We removed any irrelevant information from the dataset, ensuring that it only contained data that was relevant for text classification.

- `Fine-tuning Romanian BERT`: We fine-tuned the [Romanian BERT](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1) model using the training data to improve its performance on the downstream task.

- Adjusted the `Cross Entropy  loss` based on the class weights: We tackled the problem of the `imbalanced` dataset by adjusting the cross-entropy loss function based on the `weights` of each category. This allowed us to give more weight to underrepresented categories and improve the overall performance of our model.

## Results
Our team achieved `4th` place out of `46 teams` in the competition, obtaining a `56.84% balanced accuracy` on the private test. This was the chosen metric for this competition. Our approach proved to be effective in achieving a high level of accuracy on this challenging task.

## Other Attempts
We have also attempted to improve our results through `ensemble` techniques and `backtranslation`.

- However, we found that our approach using fine-tuned Romanian BERT with adjusted CE loss provided the best results.

- Regarding backtranslation, although we tried to augment the dataset, we encountered difficulties due to the nature of the language being sexist and offensive. The back-translated phrases did not contain bad words, which resulted in limited improvement.

## Future Approaches
- We believe that further improvements could be made by `better sanitizing` the dataset

- We believe that using a model that has been specifically trained on `similar types of text` could be beneficial. 

- Additionally, one can also explore the possibility of using `data augmentation` to improve the results. An approach similar to [Easy Data Augmentation](https://arxiv.org/abs/1901.11196) could be implemented to evaluate its effectiveness.

## [TODO]: Repository Structure

## [TODO]: Getting Started

## Team Members
- [Andrei Dumitrescu](https://github.com/AndreiDumitrescu99)
- [Cosmin Ciocan](https://github.com/cosminc98)
- [Tăiatu Iulian](https://github.com/Iulian277)

## Contact
If you have any questions about our approach or our code, please feel free to contact us at <iulian27_marius@yahoo.com>.