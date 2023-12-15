# LLM-grammar

This project about LLMs and their grammar representations is part of the Yale course CPSC488/588 "AI Foundation Models". The goal is to use a state-of-the-art LLM to augment the examples in the [EGP collection of grammar constructs](https://www.englishprofile.org/english-grammar-profile/egp-online), annotated with difficulty levels. The augmented dataset is used to train grammar detection feed forward networks on sentence embeddings generated by [ember-v1](https://huggingface.co/llmrails/ember-v1). They can be used to create a preference dataset or do rejection sampling when generating text.

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

### Data augmentation

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

### Evaluate texts with the classification models