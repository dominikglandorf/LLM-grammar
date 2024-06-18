# Towards Fine-Grained Pedagogical Control over English Grammar Complexity in Educational Text Generation

This project about using LLMs for grammar-controlled educational text generation originated in the Yale course CPSC488/588 "AI Foundation Models" by Arman Cohan. In collaboration with Detmar Meurers, it was submitted to the [BEA Workshop at NAACL 2024](https://sig-edu.org/bea/2024). The goal is to use GPT models to generate examples for the [English Grammar Profile collection of grammar constructs](https://www.englishprofile.org/english-grammar-profile/egp-online). The augmented dataset is used to train BERT-based grammar detectors. The grammar detectors are used to re-rank sentence candidates from Mistral-7B-Instruct-v0.2 by the complexity of the used grammar to make educational text appropriate for learners with differing language proficiency.

[Link to the Paper](https://github.com/dominikglandorf/LLM-grammar/blob/main/doc/Paper.pdf)

# Data
The project provides 946K generated example sentences for all 1,222 EGP entries (at least 500 positives and 250 negatives). Note that these are automatically generated. The quality estimate revealed that 87.1% can be assumed to be correctly labeled. The entire dataset can be found in the main directory under `EGP_examples.json`.

# Models
There are six trained classifcation models that need to be in the folder `models/classifiers`, one per CEFR level. You can download them from [Google Drive](https://drive.google.com/drive/folders/1irw6tERfxQP8j0dtZvd4DE2xHpCMJAue?usp=sharing). For an example on how to use them, see the functions `load_model` and `get_scores` in `/source/models.py`.

# Running the experiments and scripts

## Prerequisites

- Ensure you have `conda` installed on your machine. 

### Installing the environment

1. To install the Conda environment for this project, run the following command:
```bash
conda env create -f environment.yml
```

2. Before running any scripts, ensure you activate the environment with:
```bash
conda activate llm
```

3. For using spacy in experiments 8-10, please execute first
```bash
python -m spacy download en_core_web_sm
```

4. If you want to generate text using the Google Cloud, install the gcloud CLI following their [guide](https://cloud.google.com/sdk/docs/install).

### Configuring the environment

1. Create a copy of `config.py.example` named `config.py` and insert your API keys and path to the gcloud credentials of a service account with the appropriate rights.

## Execute scripts

### Experiments

You can reproduce the chronologically enumerated experiments by executing the Jupyter notebooks in the folder `exp`. However, it is suggested for reproduction to use the scripts below.

### Example generation

To create the augmented dataset, execute the Python script `generate_examples.py` in the directory `src` with the following options:
```
--examples-per-batch EXAMPLES_PER_BATCH
                    Positive and negative examples per batch (default: 20)
--batches BATCHES     Batches (default: 5)
--samples-per-level SAMPLES_PER_LEVEL
                    Samples per CEFR level (default: 1)
--input-file INPUT_FILE
                    Name of input file in folder dat (default: egponline.csv)
--output-file OUTPUT_FILE
                    Name of output file in folder dat (default: egpaugmented.json)
```

### Training the classification models

For the training of the classifiers, execute the Python script `train_classifiers.py` in the directory `src` with the following options:
```
  --input-file INPUT_FILE
                        Name of input file in folder dat (default: egpaugmented.csv)
  --output-dir OUTPUT_DIR
                        Name of output directory for model checkpoints (default: models)
```