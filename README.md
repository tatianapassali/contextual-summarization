# Contextual Abstractive Summarization 

This repository contains code for paper: [Controllable Abstractive Summarization with Arbitrary Textual Context](https://doi.org/10.1017/nlp.2025.10009)

## Requirements
`nltk==3.5`\
`numpy==1.22.4`\
`pandas==1.4.2`\
`scikit_learn==1.3.2`\
`torch==1.12.0`\
`transformers==4.21.1`\

## Installation 
Clone the repository locally and 
install the necessary library dependencies from requirements.txt
```
$ git clone https://github.com/tatianapassali/contextual-summarization/
$ cd contextual-summarization
$ pip3 install -r requirements.txt
```

Alternatively, you can create a python virtual environment (venv) using the virtualenv tool.
Just make sure that you run Python 3.8 or more. After cloning the repository, as shown above,
you have to initialize and activate the virtual enviroment.
```
$ cd contextual-summarization/
$ virtualenv contextual-summarization
$ source contextual-summarization//bin/activate
$ pip3 install -r requirements.txt
```
### Generate data with tags 
To generate tagged data, use the main file with the following argument:

* `input_file`: Path to the input CSV file that contains a text column to assign the tag tokens.
* `output_file`: Path to save the output CSV file with the tag tokens.
* `chunksize`: Batch of records for preprocessing.
* `text_column`: The text column that needs to be tagged.
* `guidance_column`: The column that contains textual information to guide the text for tagging.

```python3 generate_data.py \
    --input_file ./input.csv \
    --output_file ./output.csv \
    --chunksize 1000 \
    --text_column text \
    --guidance_column guidance
```

## Get embeddings representations
We provide the embedding representations of the topics in the vox datasets in the vox directory :

* label_embeddings: Embeddings of each single topic.
* collection_embeddings: Embeddings of collections of documents for each topic.

## Fine-tune models
You can fine-tune your own models using the existing tagged datasets or generate tags on your own dataset. To fine-tune your own model with tagging, you can run the run.summarization.py as follows:

```
python3 run_summarization.py \
    --model_name_or_path MODEL_NAME \
    --tokenizer MODEL_NAME \
    --do_train \
    --do_eval \
    --train_file PATH_OF_TRAIN_FILE \
    --test_file  PATH_OF_TEST_FILE \
    --validation_file PATH_OF_VALIDATION_FILE \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=12 \
    --predict_with_generate \
```

### Measure trustworthiness
You can  measure the trustworthiness of the generated summaries as follows:
```
analyzer = DocumentTrustworthinessAnalyzer(string_topics, file_path, data_path, topic_path)
analyzer.analyze_trustworthiness()
```

## License 
This project is released under Apache 2.0 license.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
