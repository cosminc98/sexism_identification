<div align="center">

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
<br>
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<br>
[![tests](https://github.com/cosminc98/sexism_identification/actions/workflows/test.yml/badge.svg)](https://github.com/cosminc98/sexism_identification/actions/workflows/test.yml)
[![code-quality](https://github.com/cosminc98/sexism_identification/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/cosminc98/sexism_identification/actions/workflows/code-quality-main.yaml)
<br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/cosminc98/sexism_identification#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/cosminc98/sexism_identification/pulls)
[![contributors](https://img.shields.io/github/contributors/cosminc98/sexism_identification.svg)](https://github.com/cosminc98/sexism_identification/graphs/contributors)

</div>

<br>

# Text Classification in Romanian Language

This repository contains the code for our team's submission to a Natural Language Processing (NLP) competition ([Nitro](https://www.kaggle.com/competitions/nitro-language-processing-2)) hosted on Kaggle . The competition challenged participants to develop a pipeline for sexism text identification in the Romanian language.

## Competition Details

The task in this competition was to classify each text into one of `five` possible categories: (0) `Sexist Direct`, (1) `Sexist Descriptive`, (2) `Sexist Reporting`, (3) `Non-sexist Offensive`, and (4) `Non-sexist Non-offensive`.

The data for this competition has been collected from a variety of sources, including social media networks such as Facebook, Twitter, and Reddit, web articles, and books.

## Disclaimer



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

## \[TODO\]: Repository Structure

## Getting Started

### Predict with a pretrained model
Keep in mind that a prediction will be made for each line of text.
```bash
# predict using text from a file
cat data/ro/predict_example.txt | python src/predict.py --models cosminc98/sexism-identification-coroseof

# see the results
cat predictions/prediction.tsv

# predict from stdin; after entering the command write however many sentences
# you want and end with the with the [EOF] marker:
#   Femeile au locul în bucătărie
#   [EOF]
python src/predict.py --models cosminc98/sexism-identification-coroseof
```

### Training a new model
```bash
# run a single training run with CoRoSeOf dataset and default hyperparameters
python src/train.py

# run hyperparameter search with the Optuna plugin from Hydra
python src/train.py -m hparams_search=optuna
```

### Creating a new Kaggle Submission
```bash
# predict on the test set
python src/test.py --models cosminc98/sexism-identification-coroseof
```
Now all you need to do is [upload](https://www.kaggle.com/competitions/nitro-language-processing-2/submissions) "submissions/submission.csv" to Kaggle.

### Uploading to Hugging Face

A pretrained model in the Romanian language is already available [on huggingface](https://huggingface.co/cosminc98/sexism-identification-coroseof).

```bash
pip install huggingface_hub

# add your write token using
huggingface-cli login

echo "
push_to_hub: True
hub_model_id: \"<model-name>\"
" >> configs/trainer/default.yaml
```

## Team Members

- [Andrei Dumitrescu](https://github.com/AndreiDumitrescu99)
- [Cosmin Ciocan](https://github.com/cosminc98)
- [Tăiatu Iulian](https://github.com/Iulian277)

## Contact

If you have any questions about our approach or our code, please feel free to contact us at <iulian27_marius@yahoo.com>.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cosminc98"><img src="https://avatars.githubusercontent.com/u/57830279?v=4?s=100" width="100px;" alt="Ștefan-Cosmin Ciocan"/><br /><sub><b>Ștefan-Cosmin Ciocan</b></sub></a><br /><a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=cosminc98" title="Code">💻</a> <a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=cosminc98" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Iulian277"><img src="https://avatars.githubusercontent.com/u/31247431?v=4?s=100" width="100px;" alt="Iulian Taiatu"/><br /><sub><b>Iulian Taiatu</b></sub></a><br /><a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=Iulian277" title="Code">💻</a> <a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=Iulian277" title="Documentation">📖</a> <a href="#research-Iulian277" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreiDumitrescu99"><img src="https://avatars.githubusercontent.com/u/65347453?v=4?s=100" width="100px;" alt="AndreiDumitrescu99"/><br /><sub><b>AndreiDumitrescu99</b></sub></a><br /><a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=AndreiDumitrescu99" title="Code">💻</a> <a href="#research-AndreiDumitrescu99" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!