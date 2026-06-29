import numpy as np # linear algebra
import pandas as pd #


from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random
import pickle
import matplotlib.pyplot as plt
from Bio import SeqIO
from torch import Tensor, tensor, float32
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_data(correctly_assigned_reads_filename, incorrectly_assigned_reads_filename):
    records_correct = list(SeqIO.parse(correctly_assigned_reads_filename, "fasta"))
    records_incorrect = list(SeqIO.parse(incorrectly_assigned_reads_filename, "fasta"))

    i = 0
    #sequences = np.array(['' for i in range(len(records))])
    sequences = ['' for i in range(len(records_correct) + len(records_incorrect))]

    for record in records_correct:
        sequences[i] = str(record.seq)
        i += 1


    for record in records_incorrect:
        sequences[i] = str(record.seq)
        i += 1

    print(sequences[:5])
    print(len(sequences))
    print("################################# ")
    #sequence_tensor = Tensor(sequences)
    labels = Tensor([1 for j in range(len(records_correct))]+ [0 for k in range(len(records_incorrect))])
    return sequences, labels

#incorrect_filename = 'test_genomic_data/incorrect_BA.fasta'
#correct_filename = 'test_genomic_data/correct_BA.fasta'
correct_filename = "/home/katie/Documents/misclassified_read_classifier/correct_BA_2k.fasta"
incorrect_filename = "/home/katie/Documents/misclassified_read_classifier/misclassified_ba_2k.fasta"
input_data, labels = load_data(correct_filename, incorrect_filename)



def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


def getKmers(sequence, size=8):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]



def test_classifier(x_train, y_train, x_test, y_test, ClassifierClass, **kwargs):
    classifier = ClassifierClass(**kwargs)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Confusion matrix for predictions of misclassified BA DNA sequence\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    return (accuracy, precision, recall, f1)



## make function for input

def load_data_make_kmers(incorrect_filename, correct_filename, k=6):
    input_data, labels = load_data(correct_filename, incorrect_filename)
    input_df = pd.DataFrame({"Sequence": input_data, "Class": labels})
    input_df['words']=input_df.apply(lambda x: getKmers(x['Sequence'],size=6), axis=1)
    
    # words is list of kmers
    texts = list(input_df['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(texts[item])
    #separate labels, this is their y_human
    y_labels = input_df["Class"].values 


    cv = CountVectorizer(ngram_range=(4,4)) #The n-gram size of 4 is previously determined by testing
    X = cv.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y_labels, 
                                                        test_size = 0.20, 
                                                        random_state=42) 
    return (X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = load_data_make_kmers(incorrect_filename, correct_filename)


from sklearn.naive_bayes import GaussianNB

test_gaussian_classified = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, GaussianNB, var_smoothing=1e-10)

from sklearn.naive_bayes import MultinomialNB

test_gaussian_classified = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MultinomialNB, alpha=0.1)

test_gaussian_classified = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MultinomialNB, alpha=0.2)
# varying alpha doesn't change it much, making much smaller makes f1 slightly lower

test_gaussian_classified = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, GaussianNB, var_smoothing=1e-11)
# varying var smoothing doesn't chaznge much

X_train_7, X_test_7, y_train_7, y_test_7 = load_data_make_kmers(incorrect_filename, correct_filename, 7)

test_gaussian_classified = test_classifier(X_train_7.toarray(), y_train_7, X_test_7.toarray(), y_test_7, GaussianNB, var_smoothing=1e-10)

## predictions don't change, think i need bigger dataset

# try with 2k of each reads
#correct_filename = "/home/katie/Documents/misclassified_read_classifier/correct_BA_2k.fasta"
correct_filename = "/home/katie/Documents/misclassified_read_classifier/correctly_classified_cp_4k.fasta"

#incorrect_filename = "/home/katie/Documents/misclassified_read_classifier/misclassified_ba_2k.fasta"
incorrect_filename = "/home/katie/Documents/misclassified_read_classifier/misclassified_cp_4k.fasta"

X_train, X_test, y_train, y_test = load_data_make_kmers(incorrect_filename, correct_filename)
## better, not surprising

# try neural network
test_MLPClassifier = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MLPClassifier, solver='lbfgs', alpha=1e-5,)

#Confusion matrix for predictions of misclassified BA DNA sequence

# Predicted  0.0  1.0
# Actual             
# 0.0        321   57
# 1.0         31  391
# accuracy = 0.890 
# precision = 0.891 
# recall = 0.890 

test_MLPClassifier = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MLPClassifier, solver='lbfgs', alpha=1e-6,)

# Confusion matrix for predictions of misclassified BA DNA sequence

# Predicted  0.0  1.0
# Actual             
# 0.0        326   52
# 1.0         31  391
# accuracy = 0.896 
# precision = 0.897 
# recall = 0.896 
# f1 = 0.896

test_MLPClassifier = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MLPClassifier, solver='lbfgs', alpha=1e-6, hidden_layer_sizes = 15)


# Predicted  0.0  1.0
# Actual             
# 0.0        328   50
# 1.0         53  369
# accuracy = 0.871 
# precision = 0.871 
# recall = 0.871 
# f1 = 0.871

## much slower
test_MLPClassifier = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MLPClassifier, solver='lbfgs', alpha=1e-6, hidden_layer_sizes = 200)

# Predicted  0.0  1.0
# Actual             
# 0.0        325   53
# 1.0         37  385
# accuracy = 0.887 
# precision = 0.888 
# recall = 0.887 
# f1 = 0.887

test_MLPClassifier = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MLPClassifier, alpha=1e-6, hidden_layer_sizes = 200)

# better
#Predicted  0.0  1.0
#Actual             
#0.0        319   59
#1.0         19  403
#accuracy = 0.902 
#precision = 0.906 
#recall = 0.902 
#f1 = 0.902

################################################################
#################################################################

compare_metrics = pd.DataFrame(columns = ["Classifier name", "alpha", "solver", "hidden layer sizes", "accuracy", "precision", "recall" ])
alphas = [1e-4, 1e-7, 1e-10, 1e-13]
solvers = ["adam", "sgd", "lbfgs"]
hidden_layer_sizes = [5, 10, 50, 100, 200]

for a in alphas:
    for s in solvers:
        for h in hidden_layer_sizes:
            accuracy, precision, recall, f1 = test_classifier(X_train.toarray(), y_train, X_test.toarray(), y_test, MLPClassifier, solver=s, alpha=a, hidden_layer_sizes = h)
            res = {"Classifier name":["MLP"], "alpha":[a], "solver": [s], "hidden layer sizes":[h] , "accuracy":[accuracy], "precision":[precision], "recall":[recall], "f1": [f1] }
            compare_metrics = pd.concat([pd.DataFrame(res), compare_metrics])


compare_metrics.to_csv("MLP_metrics_comparison_CP.csv")

## try 
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  

# Don't cheat - fit only on training data
scaler.fit(X_train.toarray())  
X_train_scaled = scaler.transform(X_train.toarray())  
# apply same transformation to test data
X_test_scaled = scaler.transform(X_test.toarray())  


for a in alphas:
    for s in solvers:
        for h in hidden_layer_sizes:
            accuracy, precision, recall, f1 = test_classifier(X_train_scaled, y_train, X_test_scaled, y_test, MLPClassifier, solver=s, alpha=a, hidden_layer_sizes = h)
            res = {"Classifier name":["MLP scaled"], "alpha":[a], "solver": [s], "hidden layer sizes":[h] , "accuracy":[accuracy], "precision":[precision], "recall":[recall], "f1": [f1] }
            compare_metrics = pd.concat([compare_metrics, pd.DataFrame(res)])


compare_metrics.to_csv("MLP_metrics_comparison_CP.csv")

X_train_5, X_test_5, y_train_5, y_test_5 = load_data_make_kmers(incorrect_filename, correct_filename, 7)

for a in alphas:
    print(f"Alpha: {a}")
    for s in solvers:
        print(f"Solver: {s}")
        for h in hidden_layer_sizes:
            print(f"Hidden layers: {h}")
            accuracy, precision, recall, f1 = test_classifier(X_train_5.toarray(), y_train_5, X_test_5.toarray(), y_test_5, MLPClassifier, solver=s, alpha=a, hidden_layer_sizes = h)
            res = {"Classifier name":"MLP K 5", "alpha":[a], "solver": [s], "hidden layer sizes":[h] , "accuracy":[accuracy], "precision":[precision], "recall":[recall], "f1": [f1] }
            compare_metrics = pd.concat([pd.DataFrame(res), compare_metrics])

compare_metrics.to_csv("MLP_metrics_comparison_CP.csv")


scaler.fit(X_train_5.toarray())  
X_train_scaled = scaler.transform(X_train_5.toarray())  
# apply same transformation to test data
X_test_scaled = scaler.transform(X_test_5.toarray())  


for a in alphas:
    print(f"Alphe: {a}")
    for s in solvers:
        print(f"Solver: {s}")
        for h in hidden_layer_sizes:
            print(f"Hidden layers: {h}")
            accuracy, precision, recall, f1 = test_classifier(X_train_scaled, y_train_5, X_test_scaled, y_test_5, MLPClassifier, solver=s, alpha=a, hidden_layer_sizes = h)
            res = {"Classifier name":["MLP scaled K 5"], "alpha":[a], "solver": [s], "hidden layer sizes":[h] , "accuracy":[accuracy], "precision":[precision], "recall":[recall], "f1": [f1] }
            compare_metrics = pd.concat([compare_metrics, pd.DataFrame(res)])

compare_metrics.to_csv("MLP_metrics_comparison_CP.csv")

compare_metrics = pd.DataFrame(columns = ["Classifier name", "alpha", "solver", "hidden layer sizes", "accuracy", "precision", "recall" ])
alphas = [1e-4, 1e-7, 1e-10, 1e-13]
solvers = ["adam", "sgd", "lbfgs"]
hidden_layer_sizes = [5, 10, 50, 100, 200]


X_train_7, X_test_7, y_train_7, y_test_7 = load_data_make_kmers(incorrect_filename, correct_filename, 7)

for a in alphas:
    print(f"Alphe: {a}")
    for s in solvers:
        print(f"Solver: {s}")    
        for h in hidden_layer_sizes:
            print(f"Hidden layers: {h}")
            accuracy, precision, recall, f1 = test_classifier(X_train_7.toarray(), y_train_7, X_test_7.toarray(), y_test_7, MLPClassifier, solver=s, alpha=a, hidden_layer_sizes = h)
            res = {"Classifier name":["MLP K 7"], "alpha":[a], "solver": [s], "hidden layer sizes":[h] , "accuracy":[accuracy], "precision":[precision], "recall":[recall], "f1": [f1] }
            compare_metrics = pd.concat([pd.DataFrame(res), compare_metrics])

compare_metrics.to_csv("MLP_metrics_comparison.csv")


scaler.fit(X_train_7.toarray())  
X_train_scaled = scaler.transform(X_train_7.toarray())  
# apply same transformation to test data
X_test_scaled = scaler.transform(X_test_7.toarray())  


for a in alphas:
    print(f"Alphe: {a}")

    for s in solvers:
        print(f"Solver: {s}")  
        for h in hidden_layer_sizes:
            print(f"Hidden layers: {h}")
            accuracy, precision, recall, f1 = test_classifier(X_train_scaled, y_train_7, X_test_scaled, y_test_7, MLPClassifier, solver=s, alpha=a, hidden_layer_sizes = h)
            res = {"Classifier name":["MLP scaled K 7"], "alpha":[a], "solver": [s], "hidden layer sizes":[h] , "accuracy":[accuracy], "precision":[precision], "recall":[recall], "f1": [f1] }
            compare_metrics = pd.concat([compare_metrics, pd.DataFrame(res)])

compare_metrics.to_csv("MLP_metrics_comparison.csv")

# so far best is, K 6
# MLP,1e-10,sgd,100,0.90625,0.90968916169159,0.90625,0.9057706705459233

## its best with K6

## try with pytorch now

# ON BA
# k7 best:
# 0,MLP scaled K 7,0.0001,sgd,50,0.91,0.9170588235294119,0.91,0.9092444749014905
#K6 best:
# Classifier name,alpha,solver,hidden layer sizes,
# 0,MLP scaled,1e-10,sgd,10,0.9125,0.9185394274493592,0.9125,0.911848426082603
# alpha = 1e10, hidden layer sizes = 10, multilayer perceptron
#     - 'sgd' refers to stochastic gradient descent.



