from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel, CLIPTextModel, CLIPTokenizer
import torch
from sklearn.cluster import KMeans
import numpy as np
from dataset_utils import relation_class_by_freq
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
    kmeans = KMeans(n_clusters=n_clusters, random_state=4).fit(embeddings)
    # Create a dictionary to map original index to cluster
    cluster_assignment = {i: cluster for i, cluster in enumerate(kmeans.labels_)}

    # Sort relation classes by cluster and create a new index map
    sorted_relations = sorted(enumerate(relation_names), key=lambda x: cluster_assignment[x[0]])

    # Create a new index map (as a tensor) that maps original index to new sorted index
    new_index_map = torch.zeros(len(relation_names), dtype=torch.long)
    for new_idx, (original_idx, _) in enumerate(sorted_relations):
        new_index_map[original_idx] = new_idx

    # Map each relation name to its cluster center
    cluster_map = {i: [relation_classes[key] for key in relation_classes.keys() if cluster_assignment[key] == i] for i in range(n_clusters)}
    return cluster_map, new_index_map


# Perform clustering and create index maps for each set of embeddings
relation_names = list(relation_classes.values())
n_clusters = 3

gpt2_cluster_map, gpt2_index_map = cluster_and_map(gpt2_embeddings, relation_names, n_clusters)
bert_cluster_map, bert_index_map = cluster_and_map(bert_embeddings, relation_names, n_clusters)
clip_cluster_map, clip_index_map = cluster_and_map(clip_embeddings, relation_names, n_clusters)
print('gpt2_cluster_map', [len(gpt2_cluster_map[key]) for key in gpt2_cluster_map.keys()], gpt2_cluster_map, '\ngpt2_index_map', gpt2_index_map, '\n\n',
      'bert_cluster_map', [len(bert_cluster_map[key]) for key in bert_cluster_map.keys()], bert_cluster_map, '\nbert_index_map', bert_index_map, '\n\n',
      'clip_cluster_map', [len(clip_cluster_map[key]) for key in clip_cluster_map.keys()], clip_cluster_map, '\nclip_index_map', clip_index_map)


# Function to plot t-SNE for given embeddings
def plot_tsne_embeddings(gpt2_emb, bert_emb, clip_emb, relation_classes):
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=5)
    gpt2_tsne = tsne.fit_transform(gpt2_emb)
    bert_tsne = tsne.fit_transform(bert_emb)
    clip_tsne = tsne.fit_transform(clip_emb)

    # Dummy cluster assignments for illustration (replace with actual clustering method)
    clusters = np.random.randint(0, 3, len(relation_classes))

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot for GPT-2
    for i, label in enumerate(relation_classes.values()):
        axes[0].scatter(gpt2_tsne[i, 0], gpt2_tsne[i, 1], c=['r', 'g', 'b'][clusters[i]], label=label if i == 0 else "")
        axes[0].text(gpt2_tsne[i, 0], gpt2_tsne[i, 1], label, fontsize=9)
    axes[0].set_title('GPT-2 Embeddings')

    # Plot for BERT
    for i, label in enumerate(relation_classes.values()):
        axes[1].scatter(bert_tsne[i, 0], bert_tsne[i, 1], c=['r', 'g', 'b'][clusters[i]], label=label if i == 0 else "")
        axes[1].text(bert_tsne[i, 0], bert_tsne[i, 1], label, fontsize=9)
    axes[1].set_title('BERT Embeddings')

    # Plot for CLIP
    for i, label in enumerate(relation_classes.values()):
        axes[2].scatter(clip_tsne[i, 0], clip_tsne[i, 1], c=['r', 'g', 'b'][clusters[i]], label=label if i == 0 else "")
        axes[2].text(clip_tsne[i, 0], clip_tsne[i, 1], label, fontsize=9)
    axes[2].set_title('CLIP Embeddings')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig('tsne_embeddings.png')

plot_tsne_embeddings(gpt2_embeddings, bert_embeddings, clip_embeddings, relation_classes)