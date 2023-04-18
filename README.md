<div align="center">

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
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 30%;">
  </picture>
</p>

</div>

<br>

# Text Classification in Romanian Language

This repository contains the code for our team's submission to a Natural Language Processing (NLP) competition ([Nitro](https://www.kaggle.com/competitions/nitro-language-processing-2)) hosted on Kaggle. The competition challenged participants to develop a pipeline for sexism text identification in the Romanian language.

## Competition Details

The task in this competition was to classify each text into one of `five` possible categories: (0) `Sexist Direct`, (1) `Sexist Descriptive`, (2) `Sexist Reporting`, (3) `Non-sexist Offensive`, and (4) `Non-sexist Non-offensive`.

- Sexist:
  - Direct: The post contains sexist elements and is directly addressed to a specific gender, usually women.
  - Descriptive: The post describes one or more individuals, usually a woman or a group of women, in a sexist manner without directly addressing them.
  - Reporting: The post reports a witnessed or heard sexist act from other sources.
- Non-sexist:
  - Offensive: The post does not contain sexist connotations but includes offensive language.
  - Non-offensive: There are no sexist or offensive elements in the post.

The data for this competition has been collected from a variety of sources, including social media networks such as Facebook, Twitter, and Reddit, web articles, and books.

## Disclaimer
The data used in this project includes instances of sexism and hate speech Therefore, reader discretion is strongly advised. The contributors to this project strongly oppose discrimination based on gender, religion, race or any other kind. One of the goals of this project is to raise awareness about gender bias online.

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

<table>
<thead>
  <tr>
    <th>Ground Truth</th>
    <th>Predicted</th>
    <th>Text</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>non-offensive</td>
    <td>descriptive</td>
    <td>Bărbații sunt de obicei mai înalți decât femeile, în medie</td>
  </tr>
  <tr>
    <td>non-offensive</td>
    <td>non-offensive</td>
    <td>Toate genurile merită să se simtă în siguranță în comunitățile lor</td>
  </tr>
  <tr>
    <td>non-offensive</td>
    <td>non-offensive</td>
    <td>Am o nevastă și trei copii.</td>
  </tr>
  <tr>
    <td>offensive</td>
    <td>offensive</td>
    <td>Sa mi bag pula in toate numerel de inmatriculare de pe planeta asta</td>
  </tr>
  <tr>
    <td>offensive</td>
    <td>offensive</td>
    <td>Învață bă unde sunt tastele anal-fagotule</td>
  </tr>
  <tr>
    <td>offensive</td>
    <td>direct</td>
    <td>Sa ma sugi si de pula si de penis</td>
  </tr>
  <tr>
    <td>descriptive</td>
    <td>descriptive</td>
    <td>Care e diferența dintre o femeie și o vacă? Vacii nu se enervează când le spui că au nevoie de mai mult lapte.</td>
  </tr>
  <tr>
    <td>descriptive</td>
    <td>descriptive</td>
    <td>Bărbații sunt mai potriviți pentru poziții de putere.</td>
  </tr>
  <tr>
    <td>descriptive</td>
    <td>descriptive</td>
    <td>Femeile au locul în bucătărie</td>
  </tr>
  <tr>
    <td>direct</td>
    <td>direct</td>
    <td>Sa iti dau cu pula peste cur pana iti raman urme rosii</td>
  </tr>
  <tr>
    <td>direct</td>
    <td>direct</td>
    <td>Nu vezi fă cât ești de grasă că te scoate cu elicopterul dacă ai căzut în gaura de canalizare</td>
  </tr>
  <tr>
    <td>direct</td>
    <td>direct</td>
    <td>Sunt 20 de grade afară, dar pizda ta are mai multe grade de alcoolemie după ce am stropit-o pe față cu șampanie</td>
  </tr>
  <tr>
    <td>reporting</td>
    <td>reporting</td>
    <td>Normalizarea hărțuirii și a agresiunii sexuale, adesea prin prezentarea personajelor feminine ca fiind dispuse sau meritând un astfel de tratament</td>
  </tr>
  <tr>
    <td>reporting</td>
    <td>reporting</td>
    <td>O tanara a fost violata de catre un fost iubit.</td>
  </tr>
  <tr>
    <td>reporting</td>
    <td>descriptive</td>
    <td>Femeilor li se refuză dreptul de a deține proprietate sau de a avea controlul asupra propriilor finanțe în multe societăți</td>
  </tr>
</tbody>
</table>

As we can see in the table above, the model works most of the time but, because of the metric choice of the challenge, it tends to predict false positives (sexist or offensive instead of the most common label of non-offensive). In practice, more data would be needed and a higher threshold would be set for the decision to flag a comment for sexist or offensive content.

## Other Attempts

We have also attempted to improve our results through `ensemble` techniques and `backtranslation`.

- However, we found that our approach using fine-tuned Romanian BERT with adjusted CE loss provided the best results.

- Regarding backtranslation, although we tried to augment the dataset, we encountered difficulties due to the nature of the language being sexist and offensive. The back-translated phrases did not contain bad words, which resulted in limited improvement.


## Future Approaches

- We believe that further improvements could be made by `better sanitizing` the dataset

- We believe that using a model that has been specifically trained on `similar types of text` could be beneficial.

- Additionally, one can also explore the possibility of using `data augmentation` to improve the results. An approach similar to [Easy Data Augmentation](https://arxiv.org/abs/1901.11196) could be implemented to evaluate its effectiveness.

## Project Structure

```
.
├── .devcontainer                           <- Dev Container Setup
├── .github                                 <- Github Workflows
├── .project-root                           <- Used to identify the project root
├── .vscode                                 <- Visual Studio Code settings
├── configs                                 <- Hydra configs
│   ├── data                                    <- Dataset configs
│   ├── hparams_search                          <- Hyperparameter Search configs
│   ├── hydra                                   <- Hydra runtime configs
│   ├── model                                   <- Model configs
│   ├── paths                                   <- Commonly used project paths
│   ├── predict.yaml                            <- predict.py configs
│   ├── test.yaml                               <- test.py configs
│   ├── train.yaml                              <- train.py configs
│   └── trainer                                 <- Transformers trainer configs
├── data                                    <- Datasets
│   └── ro                                      <- Romanian language
│       ├── predict_example.txt                     <- Small sample to be used with predict.py
│       ├── test_data.csv                           <- CoRoSeOf Test Data
│       └── train_data.csv                          <- CoRoSeOf Training Data
├── experiments                             <- Experiments directory
│   └── train
│       ├── multiruns                           <- Hyperparameter search
│       └── runs                                <- Single experiments
├── notebooks
│   └── hackathon_notebook.ipynb            <- The original hackathon notebook
├── predictions                             <- Results from predict.py
├── src                                     <- Source code
│   ├── __init__.py
│   ├── data                                    <- Dataset related
│   │   └── coroseof_datamodule.py
│   ├── predict.py
│   ├── test.py
│   ├── train.log
│   ├── train.py
│   ├── trainers
│   │   └── imbalanced_dataset_trainer.py       <- Custom trainer with class weights
│   └── utils
│       └── config.py                           <- Custom Omegaconf resolvers
├── submissions                             <- Results from test.py
└── tests                                   <- Tests directory
```

## Getting Started

Thanks to our [devcontainer setup](https://containers.dev/) you can run our model right here on GitHub. Just [create a new codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace) and follow the steps below! Keep in mind, training requires a GPU, which is not available on GitHub, so you might want to use Visual Studio Code for that.

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
# the model will be available in experiments/train/runs/
python src/train.py

# run hyperparameter search with the Optuna plugin from Hydra
# the model will be available in experiments/train/multiruns/
python src/train.py -m hparams_search=optuna
```
The model

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

## Contact

If you have any questions about our approach or our code, please feel free to contact us at:
 -  <iulian27_marius@yahoo.com>
 -  <ciocan.cosmin98@gmail.com>

## Contributors ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cosminc98"><img src="https://avatars.githubusercontent.com/u/57830279?v=4?s=100" width="100px;" alt="Ștefan-Cosmin Ciocan"/><br /><sub><b>Ștefan-Cosmin Ciocan</b></sub></a><br /><a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=cosminc98" title="Code">💻</a> <a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=cosminc98" title="Documentation">📖</a><a href="#research-cosminc98" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Iulian277"><img src="https://avatars.githubusercontent.com/u/31247431?v=4?s=100" width="100px;" alt="Iulian Taiatu"/><br /><sub><b>Iulian Taiatu</b></sub></a><br /><a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=Iulian277" title="Code">💻</a> <a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=Iulian277" title="Documentation">📖</a> <a href="#research-Iulian277" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreiDumitrescu99"><img src="https://avatars.githubusercontent.com/u/65347453?v=4?s=100" width="100px;" alt="AndreiDumitrescu99"/><br /><sub><b>AndreiDumitrescu99</b></sub></a><br /><a href="https://github.com/Ștefan-Cosmin Ciocan/sexism_identification/commits?author=AndreiDumitrescu99" title="Code">💻</a> <a href="#research-AndreiDumitrescu99" title="Research">🔬</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
