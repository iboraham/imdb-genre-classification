# :movie_camera: IMDB Genre Classification

<div align="center">

![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/iboraham/imdb-genre-classification?style=for-the-badge&logo=appveyor) ![GitHub](https://img.shields.io/github/license/iboraham/imdb-genre-classification?style=for-the-badge&logo=appveyor)

</div>

<br>

This is a project to classify movie genres based on the IMDB dataset. The dataset is available at [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) by [@radmirkaz](https://www.kaggle.com/hijest).

Training and evaluation is done using [PyTorch](https://pytorch.org/). The model is a bert-based model, which is trained using [HuggingFace](https://huggingface.co/) transformers. The model is trained using [Kaggle](https://www.kaggle.com/) notebooks with 1x Tesla P100-PCIE-16GB GPU.

Used transfer learning to train the model on the IMDB dataset using pre-trained bert-base-uncased model which has 12 layers, 12 attention heads, 110M parameters. The model is trained for 3 epochs and the validation loss is 1.25 which is higher than other approaches on [kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb/code). The model is saved in the [model](./model) directory.

To track the training and evaluation, [Weights & Biases](https://wandb.ai/) is used. Check out below sections for the training and evaluation charts.

## Installation

In order to run the project, you need to install the dependencies. You can do this by running the following command:

```bash
pipenv install
```

## Usage

```bash
python inference.py # for the inference
```

Try [IPython](./docs/kaggle_train.ipynb) file for the training and evaluation.

## Training

The model is trained using [Kaggle](https://www.kaggle.com/) notebooks with 1x Tesla P100-PCIE-16GB GPU. The training and evaluation is done using [Weights & Biases](https://wandb.ai/) to track the training and evaluation. The training and evaluation charts are available below.

![Loss](./docs/loss.png)

Global step: 20k has the lowest eval_loss while training loss is still decreasing. The model is saved at this step.

## Acknowledgements

- [@radmirkaz](https://www.kaggle.com/hijest) for the [original dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

## Authors

<p>
  <a href="https://github.com/iboraham">
    <img src="https://github.githubassets.com/favicon.ico" alt="GitHub logo" width="30" height="30">
  </a>
  <a href="https://www.kaggle.com/<username>">
    <img src='./docs/kaggle.svg' alt="Kaggle logo" width="30" height="30">
  </a>
  <p>
</p>
