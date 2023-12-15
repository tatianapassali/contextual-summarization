import torch
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from numpy import loadtxt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class DocumentTrustworthinessAnalyzer:
    def __init__(self, string_topics, file_path, data_path, topic_path):
        # Check if a CUDA-compatible GPU is available, else use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1").to(self.device)

        self.string_topics = string_topics
        self.file_path = file_path
        self.data_path = data_path
        self.topic_path = topic_path

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

    def compute_cosine_document_embedding(self, document, topic):
        # Get word embeddings
        document_embeddings = self.generate_word_embeddings(document, generate_mean_document_embeddings=True)

        embeddings_from_numpy = self.load_embeddings_from_path(self.topic_path)

        topic_embeddings = embeddings_from_numpy[string_topics.index(topic)]
        topic_embeddings = np.squeeze(topic_embeddings)

        document_embeddings = np.squeeze(document_embeddings)

        similarity = cosine_similarity(document_embeddings.reshape(1, -1), topic_embeddings.reshape(1, -1))

        return similarity

    def compute_cosine_document_summary_representations(self, document, summary):
        # Get word embeddings
        document_embeddings = self.generate_word_embeddings(document, generate_mean_document_embeddings=True)

        # Get mean representation for summary
        summary_embeddings = self.generate_word_embeddings(summary, generate_mean_document_embeddings=True)

        # Compute cosine similarity
        similarity = cosine_similarity(document_embeddings, summary_embeddings)

        return similarity

    def load_embeddings_from_path(self, path):
        loaded_embeddings = np.load(path, allow_pickle=True)
        return loaded_embeddings

    def analyze_trustworthiness(self):
        self.trustworthiness = []

        predictions = self.read_predictions_txt(self.file_path)

        test_data = pd.read_csv(self.data_path)
        documents = test_data['document'].values.tolist()
        test_data['category'] = test_data.nearest_users.values.tolist()
        topics = test_data['category'].values.tolist()

        for i in tqdm(range(len(predictions))):
            splitted_predictions = sent_tokenize(predictions[i])

            summary_similarities = []
            document_similarities = []
            for pred in splitted_predictions:
                sim = self.compute_cosine_document_embedding(pred, topics[i])

                summary_similarities.append(sim)
                max_summary_sim = summary_similarities.index(max(summary_similarities))

                for sent in sent_tokenize(documents[i]):
                    doc_sim = self.compute_cosine_document_summary_representations(sent, splitted_predictions[max_summary_sim])
                    document_similarities.append(doc_sim)

                self.trustworthiness.append(max(document_similarities))

        print("Trustworthiness is: ", np.mean(self.trustworthiness))

    @staticmethod
    def read_predictions_txt(file_path):
        predictions_list = []
        incomplete_prediction = False
        with open(file_path, 'r', encoding='utf8') as file_pointer:
            line = file_pointer.readline()
            while line:
                if "PREDICTION: " in line or incomplete_prediction:
                    if "PREDICTION: " in line:
                        predictions_list.append("PREDICTION: " + line.split("PREDICTION: ")[-1])
                        incomplete_prediction = True
                    else:
                        predictions_list[-1] += line.split("PREDICTION: ")[-1]
                line = file_pointer.readline()

        parsed_list_from_file = [item.replace("PREDICTION: ", " ") for item in predictions_list]

        return parsed_list_from_file


string_topics = loadtxt("./categories.txt", delimiter="\n",
                        dtype=str, encoding='utf8').tolist()
file_path = './generated_predictions.txt'
data_path = './single_topic_test.csv'
topic_path = './label_embeddings.npy'

analyzer = DocumentTrustworthinessAnalyzer(string_topics, file_path, data_path, topic_path)
analyzer.analyze_trustworthiness()
