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
from authors_distance import *
from scipy.spatial import distance
from sklearn.decomposition import PCA
import k_mean
from sklearn.ensemble import RandomForestClassifier

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

new_node_info = {}
for i in range(len(node_info)):
    new_node_info[node_info[i][0]] = node_info[i]
node_info = new_node_info

IDs = [element[0] for element in node_info.values()]

# compute TFIDF vector of each paper
print("Training TFIDF...")
corpus = [element[5] for element in node_info.values()]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)
features_name = vectorizer.get_feature_names()
dict_word_idx = {features_name[i]: i for i in range(len(features_name))}

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

#longest title in dataset
#t = [len(element[2].split()) for element in node_info]
#max(t) # = 19

#complete node_info titles if title too short
print("Fill in missing titles...")
min_len_title = 5
percent_tfidf_to_keep = 0.8
nb_keywords_max = 10
for id, info in node_info.items():
    i = IDs.index(id)
    if len(info[2].split()) < min_len_title:
        tfidf_vector = features_TFIDF[i].toarray()[0]
        max_tfidf_weight = features_TFIDF[0].max()
        keywords_tfidf_idx = [idx for idx, weight in enumerate(tfidf_vector) if weight > percent_tfidf_to_keep * max_tfidf_weight]
        keywords_tfidf = [features_name[idx] for idx in keywords_tfidf_idx]
        if len(keywords_tfidf) > nb_keywords_max:
            keywords_tfidf = keywords_tfidf[:nb_keywords_max]
        #complete title with keywords
        info[2] + ' ' + ' '.join(keywords_tfidf)

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
print("Compute fastTesxt vectors...")
for id, info in node_info.items():
    ## abstract vectors
    abstract = info[5].split()
    abstract_vector = np.zeros(100)
    sum_weight = 0
    for j in range(len(abstract)):
        try:
            i = IDs.index(id)
            idx_column = dict_word_idx[abstract[j]]
            weight = features_TFIDF[i, idx_column]
            abstract_vector += np.array(model[abstract[j]]) * weight
            sum_weight += weight
        except KeyError:
            continue
            #print(abstract[j], "in stopwords")
    abstract_vector /= sum_weight

    abstract_fasttext_vectors[info[0]] = abstract_vector

    ## title vector
    title = info[2].split()
    title_vector = np.zeros(100)
    sum_weight = 0
    for j in range(len(title)):
        try:
            #i = IDs.index(id)
            idx_column = dict_word_idx[title[j]]
            #weight = features_TFIDF[i, idx_column]
            title_vector += np.array(model[title[j]]) #* weight
            #sum_weight += weight
        except KeyError:
            continue
            #print(title[j], "in stopwords")
    if sum_weight != 0:
        title_vector /= sum_weight
    title_fasttext_vectors[info[0]] = title_vector

    if any(np.isnan(title_vector)):
        if all(np.isnan(title_vector)):
            print('only nan value in title vector for title : ', ' '.join(title), '; and node info idx : ', i)
        else:
            print('some nan value in title vector for title : ', ' '.join(title), '; and node info idx : ', i)

df_abstract_fasttext_vectors = pd.DataFrame.from_dict(abstract_fasttext_vectors).T



print("Compute clusters..")
target_dimension = 15
number_clusters = 12
PCA_val = True
if PCA_val:
    pca = PCA(n_components=target_dimension, svd_solver='full')

    print("Fit and Transform...")
    pca.fit(df_abstract_fasttext_vectors)
    print("Transform...")
    df_abstract_fasttext_vectors_reduced = pd.DataFrame(pca.transform(df_abstract_fasttext_vectors))
    clusters, cost = k_mean(df_abstract_fasttext_vectors_reduced, number_clusters)

else:
    clusters, cost = k_mean(df_abstract_fasttext_vectors, number_clusters)


print("Load authors df...")
# training_set_reduced = training_set
# all_authors = get_authors(node_info)
# authors_df = authors_df(all_authors)

# set_authors_links(training_set_reduced, authors_df, node_info)
authors_df = pd.DataFrame.from_csv("authors_df.csv")

print("Compute number citation by text..")
num_citations_texts = {}
for info in training_set_reduced:
    source = info[0]
    target = info[1]
    citation = int(info[2])
    if citation:
        try:
            num_citations_texts[target] += 1
        except KeyError:
            num_citations_texts[target] = 1


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

# distance_tdf_idf
tfidf_diff = []

# authors_diff
authors_diff = []

citations_texts_feature = []
#citations_texts_feature_source = []

is_possible = []

#if the texts belong to the same cluster, then 1, else 0
same_cluster = []

counter = 0

#authors_df.to_csv("authors_df_full.csv")

for i in range(len(training_set_reduced)):
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]

    index_source = IDs.index(source)
    index_target = IDs.index(target)

    source_info = node_info[source]
    target_info = node_info[target]

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
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    same_cluster.append(int(clusters[index_source] == clusters[index_target]))

    abstract_cosine_distance.append(cosine_similarity(abstract_fasttext_vectors[source].reshape(1, -1),
                                                      abstract_fasttext_vectors[target].reshape(1, -1))[0][0])
    title_cosine_distance.append(
        cosine_similarity(title_fasttext_vectors[source].reshape(1, -1), title_fasttext_vectors[target].reshape(1, -1))[
            0][0])

    tfidf_diff.append(
        distance.cosine(features_TFIDF[index_source].toarray()[0], features_TFIDF[index_target].toarray()[0]))

    authors_diff += [get_authors_links(source_info[3], target_info[3], authors_df)]
    counter += 1

    is_possible += [int(source_info[1] > target_info[1])]

    try:
        num_cita_target = num_citations_texts[target]
    except KeyError:
        num_cita_target = 0
    try:
        num_cita_source = num_citations_texts[source]
    except KeyError:
        num_cita_source = 0
    citations_texts_feature.append(num_cita_source)
    #citations_texts_feature_source.append(num_cita_source)
    if counter % 1000 == True:
        print(counter, "training examples processsed")

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)



# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
training_features = np.array(
    [overlap_title, temp_diff, comm_auth, authors_diff, tfidf_diff, citations_texts_feature, #citations_texts_feature_source,
       abstract_cosine_distance, title_cosine_distance, is_possible, same_cluster]).T

# scale
training_features = preprocessing.scale(training_features)

# initialize basic SVM
# classifier = svm.LinearSVC()

# train
# classifier.fit(training_features, labels_array)

# test
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
abstract_cosine_distance_test = []
title_cosine_distance_test = []
authors_diff_test = []
tfidf_diff_test = []
citations_texts_feature_test = []
#citations_texts_feature_source_test = []
is_possible_test = []
same_cluster_test = []

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

    source_info = node_info[source]
    target_info = node_info[target]

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
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    same_cluster_test.append(int(clusters[index_source] == clusters[index_target]))
    abstract_cosine_distance_test.append(cosine_similarity(abstract_fasttext_vectors[source].reshape(1, -1), abstract_fasttext_vectors[target].reshape(1, -1))[0][0])
    title_cosine_distance_test.append(
        cosine_similarity(title_fasttext_vectors[source].reshape(1, -1), title_fasttext_vectors[target].reshape(1, -1))[
            0][0])
    authors_diff_test += [get_authors_links(source_info[3], target_info[3], authors_df)]

    try:
        num_cita_target = num_citations_texts[target]
    except KeyError:
        num_cita_target = 0
    try:
        num_cita_source = num_citations_texts[source]
    except KeyError:
        num_cita_source = 0
    citations_texts_feature_test.append(num_cita_target)
    #citations_texts_feature_source_test.append(num_cita_source)


    tfidf_diff_test.append(
        distance.euclidean(features_TFIDF[index_source].toarray()[0], features_TFIDF[index_target].toarray()[0]))

    is_possible_test += [int(source_info[1] > target_info[1])]

    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test, temp_diff_test, comm_auth_test, authors_diff_test, tfidf_diff_test,
                             citations_texts_feature_test, #citations_texts_feature_source_test,
                               abstract_cosine_distance_test, title_cosine_distance_test, is_possible_test, same_cluster_test]).T

headers = ['overlap_title', 'temp_diff', 'comm_auth', 'authors_diff', 'tfidf_diff',
                             'citations_texts_feature', # 'citations_texts_feature_source',
             'abstract_cosine_distance', 'title_cosine_distance', 'is_possible', 'same_cluster_test']


# scale
testing_features = preprocessing.scale(testing_features)
train = pd.DataFrame(training_features, columns=headers)
test = pd.DataFrame(testing_features, columns=headers)
labels = pd.DataFrame(labels_array, columns=["label"])
train.to_csv("train.csv")
test.to_csv("test.csv")
labels.to_csv("train_Y.csv")

#issue predictions
#predictions_SVM = list(classifier.predict(testing_features))

# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
# predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

classifier = svm.SVC()
classifier.fit(train.drop('label', axis=1), labels)
predictions_SVM = list(classifier.predict(test))
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)


classifier = RandomForestClassifier(max_depth=4, random_state=0)
classifier.fit(train.drop('label', axis=1), labels)
predictions_SVM = list(classifier.predict(test))
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier(n_estimators=500)
classifier.fit(train.drop('label', axis=1), labels)
predictions_SVM = list(classifier.predict(test))
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5, max_depth=3)
classifier.fit(train.drop('label', axis=1), labels)
predictions_SVM = list(classifier.predict(test))
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

clf1 = svm.SVC(probability=True)
clf2 = RandomForestClassifier(max_depth=6, n_estimators=50)
clf3 = AdaBoostClassifier(n_estimators=1000)
clf4 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=6)

eclf = VotingClassifier(estimators=[('svc', clf1), ('rf', clf2), ('ada', clf3), ('gb', clf4)], voting='soft')
params = {'rf__n_estimators': [50, 1000], 'rf__max_depth': [1, 6], 'gb__learning_rate': [0.2, 2], 'gb__max_depth': [1, 6], 'gb__n_estimators': [50, 1000], 'ada__n_estimators': [50, 1000] }

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, scoring='f1')
grid = grid.fit(train.drop('label', axis=1), labels)
predictions_SVM = list(grid.predict(test))
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

#grid.best_params_
# {'ada__n_estimators': 1000,
#  'gb__learning_rate': 0.2,
#  'gb__max_depth': 6,
#  'gb__n_estimators': 50,
#  'rf__max_depth': 6,
#  'rf__n_estimators': 50
#  }

#grid.best_score_ -> 0.9448000514855931


with open("submission_grid_search.csv", "w") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)
