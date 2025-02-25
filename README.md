
Official repository for Towards weight-space interpretation of Low-Rank Adapters for Diffusion Models paper



## Prerequisites

- To run experiments, Python at least 3.9 is required.

- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running experiments

To reproduce experiments from the paper, run:

```bash
python experiment.py --dataset=1k --num-experiments=10 --output-directory=out --flat-vec --stats-flat-vec --pca-flat-vec --stats-concat --dense
```
Note that there are 4 datasets available:

- 1k - 1000 images per class
- 10k - 10000 images per class
- 50k - 50000 images per class
- 100k - 100000 images per class

The classes are evenly distributed across all datasets.

On a single A100 GPU, it takes around 10 minutes to run all the experiments for the 1k dataset. It takes a couple of hours on A100 to run experiments for 50k.
Most of the time is spent on pre-processing the columns of weights for different adapters into a flattened vector of about 100k dimensionality.


To reproduce the nsfw experiment from the paper, run:

```bash
python nsfw-experiment.py
```


### Dataset creation code

The code for creating the dataset and some other utilities are in the `dataset` folder. Here is a quick overview of the code:

- `download-big-dataset.py` - downloads the ImageNet-1k dataset and creates a json file with the class names and ids. Extract the leaf categories and builds the 10 root level categories.
- `distribute-big-dataset.py` - distributes the images into folders, each containing images of a concept to be used for training a LoRA.
- `multi_lora_train_dreambooth.py` - trains LoRAs for a given using the DreamBooth method.
- `train-loras-from-folder.sh` - trains a LoRA for a given class on ImageNet-1k.
- `generate.py` - generates images of a dog using a fine-tuned adapter and a pre-trained model.
- `build-ws-dataset.py` - builds a dataset of weight-space vectors for a given concept from the trained LoRAs.
- `check-in-dataset.py` - pushes the dataset build by `build-ws-dataset.py` to the Hugging Face Hub.

- `download-nsfw-dataset.py` - downloads the NSFW dataset from the Hugging Face Hub and prepares it for distribution.
- `distribute-nsfw-dataset.py` - distributes the NSFW dataset into folders, each containing images of a concept to be used for training a MLP.

- `slurm` - folder with SLURM scripts for running the experiments on the PLGrid cluster.
