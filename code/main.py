import random
import numpy as np
# import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
import csv
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.decomposition import PCA
import k_mean
from sklearn.metrics import silhouette_samples, silhouette_score
import collections
import matplotlib.pyplot as plt
from authors_distance import *
from scipy.spatial import distance




nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]


# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing

# the columns of the data frame below are:
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)
features_name = vectorizer.get_feature_names()
dict_word_idx = {features_name[i]: i for i in range(len(features_name))}

#longest title in dataset
#t = [len(element[2].split()) for element in node_info]
#max(t) # = 19

#complete node_info titles if title too short
min_len_title = 5
percent_tfidf_to_keep = 0.8
nb_keywords_max = 10
for i in range(len(node_info)):
    if len(node_info[i][2].split()) < min_len_title:
        tfidf_vector = features_TFIDF[i].toarray()[0]
        max_tfidf_weight = features_TFIDF[0].max()
        keywords_tfidf_idx = [idx for idx, weight in enumerate(tfidf_vector) if weight > percent_tfidf_to_keep * max_tfidf_weight]
        keywords_tfidf = [features_name[idx] for idx in keywords_tfidf_idx]
        if len(keywords_tfidf) > nb_keywords_max:
            keywords_tfidf = keywords_tfidf[:nb_keywords_max]
        #complete title with keywords
        node_info[i][2] + ' ' + ' '.join(keywords_tfidf)

## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

# edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

# nodes = IDs

## create empty directed graph
# g = igraph.Graph(directed=True)

## add vertices
# g.add_vertices(nodes)

## add edges
# g.add_edges(edges)

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.05)))
training_set_reduced = [training_set[i] for i in to_keep]


# full_text = ''
# for i in range(len(corpus)):
#    full_text += corpus[i] + ' '

# file = open('full_text.txt', 'w')
# file.write(full_text)
# file.close()
model = fasttext.load_model('model.bin')
abstract_fasttext_vectors = {}
title_fasttext_vectors = {}
for i in range(len(node_info)):

    ## abstract vectors

    abstract = node_info[i][5].split()
    abstract_vector = np.zeros(100)
    sum_weight = 0
    for j in range(len(abstract)):
        try:
            idx_column = dict_word_idx[abstract[j]]
            weight = features_TFIDF[i, idx_column]
            abstract_vector += np.array(model[abstract[j]]) * weight
            sum_weight += weight
        except KeyError:
            continue
            #print(abstract[j], "in stopwords")
    abstract_vector /= sum_weight

    abstract_fasttext_vectors[node_info[i][0]] = abstract_vector

    ## title vector
    title = node_info[i][2].split()
    title_vector = np.zeros(100)
    sum_weight = 0
    for j in range(len(title)):
        try:
            idx_column = dict_word_idx[title[j]]
            #weight = features_TFIDF[i, idx_column]
            title_vector += np.array(model[title[j]]) #* weight
            #sum_weight += weight
        except KeyError:
            continue
            #print(title[j], "in stopwords")
    if sum_weight != 0:
        title_vector /= sum_weight
    title_fasttext_vectors[node_info[i][0]] = title_vector

    if any(np.isnan(title_vector)):
        if all(np.isnan(title_vector)):
            print('only nan value in title vector for title : ', ' '.join(title), '; and node info idx : ', i)
        else:
            print('some nan value in title vector for title : ', ' '.join(title), '; and node info idx : ', i)


df = pd.DataFrame.from_dict(abstract_fasttext_vectors)


def compute_fasttext_vectors_kmeans_nb_clusters_opt(df):
    PCA_nb_dimensions = 15
    list_nb_clusters = []
    silhouette_scores = []
    for nb_clusters in range(2, 20, 2):
        pca = PCA(n_components=PCA_nb_dimensions, svd_solver='full')
        print("Fit and Transform...")
        pca.fit(df)
        print("Transform...")
        df_reduced = pd.DataFrame(pca.transform(df))
        #print(df)

        clusters, cost = k_mean(df_reduced, nb_clusters)
        clusters = collections.OrderedDict(sorted(clusters.items()))
        score = silhouette_score(df_reduced, list(clusters.values()), metric='cosine')
        list_nb_clusters += [nb_clusters]
        silhouette_scores += [score]
        print('Silhouette score for ', nb_clusters, ' clusters : ', score)

    plt.plot(list_nb_clusters, silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    #plt.show()
    plt.savefig('Silhouettes_kmeans_2.png')

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
def compute_tfidf_vectors_kmeans_nb_clusters_opt(df):
    truncated_svd_nb_dimensions = 15
    list_nb_clusters = []
    silhouette_scores = []
    for nb_clusters in range(2, 20, 2):
        svd = TruncatedSVD(n_components=truncated_svd_nb_dimensions)
        print("Fit and Transform...")
        svd.fit(features_TFIDF)
        print("Transform...")
        df_reduced = pd.DataFrame(svd.transform(features_TFIDF))
        # print(df)

        clusters, cost = k_mean(df_reduced, nb_clusters)
        clusters = collections.OrderedDict(sorted(clusters.items()))
        score = silhouette_score(df_reduced, list(clusters.values()), metric='cosine')
        list_nb_clusters += [nb_clusters]
        silhouette_scores += [score]
        print('Silhouette score for ', nb_clusters, ' clusters : ', score)

    plt.plot(list_nb_clusters, silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    # plt.show()
    plt.savefig('Silhouettes_kmeans_TFIDF.png')



PCA_val = False
target_dimension = 15
number_clusters = 12
if PCA_val:
    pca = PCA(n_components=target_dimension, svd_solver='full')

    print("Fit and Transform...")
    pca.fit(df)
    print("Transform...")
    df_reduced = pd.DataFrame(pca.transform(df))
    print(df)

if df_reduced:
    clusters, cost = k_mean(df_reduced, number_clusters)
else:
    clusters, cost = k_mean(df, number_clusters)

# we will use three basic features:

# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []

# cosine distance of abstract fasttext vectors
abstract_cosine_distance = []

# cosine distance of title fasttext vectors
title_cosine_distance = []

# 1 if target was published before source (oriented citation possible); 0 otherwise
date_compatible = []

counter = 0
for i in range(len(training_set_reduced)):
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]

    index_source = IDs.index(source)
    index_target = IDs.index(target)

    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    overlap_title.append(len(set(source_title).intersection(set(target_title))))

    temp_diff.append(int(source_info[1]) - int(target_info[1]))

    date_compatible.append(int(int(source_info[1]) - int(target_info[1]) > 0))


    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    abstract_cosine_distance.append(cosine_similarity(abstract_fasttext_vectors[source].reshape(1, -1), abstract_fasttext_vectors[target].reshape(1, -1))[0][0])
    title_cosine_distance.append(cosine_similarity(title_fasttext_vectors[source].reshape(1, -1), title_fasttext_vectors[target].reshape(1, -1))[0][0])


    counter += 1
    if counter % 1000 == True:
        print(counter, "training examples processsed")

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, temp_diff, date_compatible, comm_auth, abstract_cosine_distance, title_cosine_distance]).T

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

# initialize basic SVM
classifier = svm.LinearSVC()

# train
classifier.fit(training_features, labels_array)

# test
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
abstract_cosine_distance_test = []
title_cosine_distance_test = []
date_compatible_test = []

counter = 0
for i in range(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]

    index_source = IDs.index(source)
    index_target = IDs.index(target)

    # source = training_set_reduced[i][0]
    # target = training_set_reduced[i][1]
    #
    # index_source = IDs.index(source)
    # index_target = IDs.index(target)

    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))

    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))

    date_compatible_test.append(int(int(source_info[1]) - int(target_info[1]) > 0))

    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))

    abstract_cosine_distance_test.append(cosine_similarity(abstract_fasttext_vectors[source].reshape(1, -1), abstract_fasttext_vectors[target].reshape(1, -1))[0][0])
    title_cosine_distance_test.append(cosine_similarity(title_fasttext_vectors[source].reshape(1, -1), title_fasttext_vectors[target].reshape(1, -1))[0][0])

    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test, temp_diff_test, date_compatible_test, comm_auth_test, abstract_cosine_distance_test, title_cosine_distance_test]).T

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

with open("fasttext_titles_oriented_predictions_training_set_*0.05.csv", "w") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)
