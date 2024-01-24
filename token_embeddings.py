from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel, CLIPTextModel, CLIPTokenizer
import torch
from sklearn.cluster import KMeans
import numpy as np
from dataset_utils import relation_class_by_freq


# # This function is defined in dataset_utils. It is copied here for reference.
# def relation_class_by_freq():
#     return {0: 'on', 1: 'has', 2: 'in', 3: 'of', 4: 'wearing', 5: 'near', 6: 'with', 7: 'above', 8: 'holding', 9: 'behind',
#             10: 'under', 11: 'sitting on', 12: 'wears', 13: 'standing on', 14: 'in front of', 15: 'attached to', 16: 'at', 17: 'hanging from', 18: 'over', 19: 'for',
#             20: 'riding', 21: 'carrying', 22: 'eating', 23: 'walking on', 24: 'playing', 25: 'covering', 26: 'laying on', 27: 'along', 28: 'watching', 29: 'and',
#             30: 'between', 31: 'belonging to', 32: 'painted on', 33: 'against', 34: 'looking at', 35: 'from', 36: 'parked on', 37: 'to', 38: 'made of', 39: 'covered in',
#             40: 'mounted on', 41: 'says', 42: 'part of', 43: 'across', 44: 'flying in', 45: 'using', 46: 'on back of', 47: 'lying on', 48: 'growing on', 49: 'walking in'}


# Get relation classes
relation_classes = relation_class_by_freq()

# Initialize tokenizers and models for GPT-2, BERT, and CLIP
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")


# Function to get embeddings
def get_embeddings(model, tokenizer, sentences):
    # Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else '[PAD]'

    # Tokenize input
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# Get embeddings for each model
gpt2_embeddings = get_embeddings(gpt2_model, gpt2_tokenizer, list(relation_classes.values()))
bert_embeddings = get_embeddings(bert_model, bert_tokenizer, list(relation_classes.values()))
clip_embeddings = get_embeddings(clip_model, clip_tokenizer, list(relation_classes.values()))


# Function for clustering and index mapping
def cluster_and_map(embeddings, relation_names, n_clusters):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    # Create a dictionary to map original index to cluster
    cluster_assignment = {i: cluster for i, cluster in enumerate(kmeans.labels_)}

    # Sort relation classes by cluster and create a new index map
    sorted_relations = sorted(relation_names, key=lambda x: cluster_assignment[relation_names.index(x)])
    new_index_map = {i: relation for i, relation in enumerate(sorted_relations)}

    # Map each relation name to its cluster center
    cluster_map = {i: [relation_classes[key] for key in relation_classes.keys() if cluster_assignment[key] == i] for i in range(n_clusters)}
    return cluster_map, new_index_map


# Perform clustering and create index maps for each set of embeddings
relation_names = list(relation_classes.values())
n_clusters = 3

gpt2_cluster_map, gpt2_index_map = cluster_and_map(gpt2_embeddings, relation_names, n_clusters)
bert_cluster_map, bert_index_map = cluster_and_map(bert_embeddings, relation_names, n_clusters)
clip_cluster_map, clip_index_map = cluster_and_map(clip_embeddings, relation_names, n_clusters)
print('gpt2_cluster_map', gpt2_cluster_map, '\ngpt2_index_map', gpt2_index_map, '\n\n',
      'bert_cluster_map', bert_cluster_map, '\nbert_index_map', bert_index_map, '\n\n',
      'clip_cluster_map', clip_cluster_map, '\nclip_index_map', clip_index_map)
