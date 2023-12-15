import pandas as pd
import time
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import string
import nltk

class TextualInformationProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.text_column = 'text'
        self.guidance_column = 'guidance'

        # Check if a CUDA-compatible GPU is available, else use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1").to(self.device)

        # Initialize stopwords and punctuation
        self.load_stopwords()
        self.load_punctuation()

    def load_stopwords(self):
        # Download NLTK stopwords
        nltk.download('stopwords', quiet=True)

        # Load stopwords from NLTK and a text file
        stopwords = nltk.corpus.stopwords.words('english')

        # Import stopwords from a text file
        with open('stopwords.txt', 'r') as f:
            added_stopwords = [line.strip() for line in f.readlines()]

        # Extend the list of stopwords with additional words
        self.stopwords = stopwords + added_stopwords

    def load_punctuation(self):
        # Import punctuation from a text file
        with open('punctuation.txt', 'r') as f:
            added_punctuation = f.read().strip()

        # Define punctuation
        self.punctuation = string.punctuation + added_punctuation

    def set_text_column(self, text_column):
        self.text_column = text_column
    def set_guidance_column(self, guidance_column):
        self.guidance_column = guidance_column

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def generate_word_embeddings(self, document, generate_mean_document_embeddings=False):
        encoded_input = self.tokenizer.encode_plus(document,
                                                   max_length=512,
                                                   padding=True,
                                                   truncation=True,
                                                   truncation_strategy='max_length',
                                                   return_tensors='pt').to(self.device)

        model_output = self.model(**encoded_input)
        if generate_mean_document_embeddings:
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']).detach().cpu().numpy()
            return embeddings
        else:
            tokens = self.tokenizer.tokenize(document, max_length=512, truncation=True,
                                             truncation_strategy='max_length')

            hidden_states = model_output[0]
            embeddings = []

            total_tokens_ids = [x for x in encoded_input.word_ids() if x is not None]
            unique_tokens_ids = np.unique(total_tokens_ids)

            merged_tokens = []

            for idx in unique_tokens_ids:
                idx_pos = np.where(np.array(encoded_input.word_ids()) == idx)[0]
                start, end = idx_pos[0], idx_pos[-1]
                if len(idx_pos) > 1:
                    merged_tokens.append(''.join(tokens[start - 1:end]).replace("##", ""))
                else:
                    merged_tokens.append(tokens[start - 1])
                word_embedding = torch.mean(hidden_states[:, start:end + 1, :], dim=1)
                embeddings.append(word_embedding.cpu().detach().numpy())

            embeddings = np.array(embeddings)
            return embeddings, merged_tokens

    def compute_cosine_document_textual_information(self, document, textual_information, embeddings_path=None):
        word_embeddings, tokens = self.generate_word_embeddings(document)
        word_embeddings = np.squeeze(word_embeddings)

        document_embeddings = self.generate_word_embeddings(textual_information, generate_mean_document_embeddings=True)
        similarities = []
        for word_embedding in word_embeddings:
            similarity = cosine_similarity(word_embedding.reshape(1, -1), document_embeddings.reshape(1, -1))
            similarities.append(similarity[0][0])

        return np.array(similarities).tolist(), tokens

    def define_threshold(self, similarities):
        similarities = np.array(similarities).reshape(-1, 1)

        gm = GaussianMixture(n_components=2, random_state=0).fit(similarities)
        threshold = np.mean(gm.means_)
        return threshold

    def assign_tags_with_threshold(self, row):
        article = row['text']
        tokens = row['tokens']
        similarities = row['similarities']
        threshold = row['threshold']

        new_tokens = []
        tagged_tokens = []

        for i in range(len(tokens)):
            if similarities[i] >= threshold and tokens[i] not in self.punctuation and tokens[i] not in self.stopwords and not tokens[i].isnumeric() and len(str(tokens[i])) != 1:
                new_tokens.append("[TAG]" + tokens[i])
                tagged_tokens.append(tokens[i])
            else:
                new_tokens.append(tokens[i])

        tagged_tokens_counter = len(tagged_tokens)
        tagged_text = " ".join(new_tokens)

        row['tagged_text'] = tagged_text
        row['tagged_tokens'] = tagged_tokens
        row['tagged_tokens_counter'] = tagged_tokens_counter

        return row['text'], row['tagged_tokens'], row['tagged_tokens_counter']

    def process_data(self, chunksize=100):
        start = time.time()
        for batch, data in enumerate(pd.read_csv(self.input_file, chunksize=chunksize, nrows=200)):
            print(f"Processing batch {batch} with chunksize {chunksize}")
            data[['similarities', 'tokens']] = data.apply(
                lambda x: self.compute_cosine_document_textual_information(x[self.text_column], x[self.guidance_column]), axis=1, result_type='expand')
            data['threshold'] = data['similarities'].apply(lambda x: self.define_threshold(x))
            print("Generating word embeddings \u2713")

            data[['tagged_article', 'tagged_tokens', 'tagged_tokens_counter']] = data.apply(
                lambda x: self.assign_tags_with_threshold(x), axis=1, result_type='expand')
            data.drop(['similarities', 'tokens', 'threshold'], axis=1)
            print("Extracting and assigning tag tokens \u2713")
            print(f"Appending tagged chunksize to file {self.output_file} \u2713")

            data.to_csv(self.output_file, mode='a', header=False, index=False)
            print("------------------------------------------")
            columns = data.columns
        end = time.time()
        print(f"Process finished in {end - start} secs")
        data = pd.read_csv(self.output_file, names=columns)
        data.to_csv(self.output_file, index=False)
