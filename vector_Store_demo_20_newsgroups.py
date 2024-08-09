# Requisite imports
import os, sys

try:
    from vector_Store import VectorStore, prettyfloat # Importing the VectorStore class from vector_store module
except Exception as exc:
    print( str(exc), str(f'Try adding the "vector_Store"(.py) module to YOUR programs path. Example: sys.path.insert(0, "C:\\Users\\User\\Desktop\\python\\tools\\vectordb\\" ))'))

try:
    import numpy as np  # Importing numpy for numerical operations
except:
    os.system('pip install numpy')
    import numpy as np

try:
    from sklearn.datasets import fetch_20newsgroups
except:
    os.system('python -m pip install scikit_learn==1.5.0')
    from sklearn.datasets import fetch_20newsgroups

try:
    from pprint import pprint
except:
    os.system('python -m pip install pprint')
    from pprint import pprint

try:
    import matplotlib.pyplot as plt
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt
try:
    from datetime import datetime as dt
except:
    os.system('pip install datetime')
    from datetime import datetime as dt
try:
    import time
except:
    os.system('pip install time')
    import time


# Establish a VectorStore instance
vector_store = VectorStore()  # Creating an instance of the VectorStore class

# nl = '\n'
# nl2 = '\n\n'
#############################
# MAIN LOGIC BEGIN HERE
#############################
dataset = fetch_20newsgroups(
    shuffle=True
    ,categories = ['misc.forsale']
    ,remove = ('headers', 'footers', 'quotes')
    ,random_state = 101
    )
print(f"The datatype of dataset is {type(dataset)}")
print('\nBy passing class.__repr__')
print(dir(dataset.__repr__))
print('\nBy passing class itself ')
pprint(dataset.__class__, width=120)

articles = list()
#DEBUG pprint(dir(dataset.data )) #'fetch_20_newsgroups'))
for text_line in dataset.data:
    #DEBUG article = (f'{text_line.replace('\n', '').replace("'",'').split("')")}')
    article = str(''.join(text_line).split("\n")).replace("\n'",'')
    #DEBUG pprint(article)
    articles.append(article)
#DEBUG  for article in articles:
#DEBUG      pprint(article)
my_sentences = articles # map(str, articles)
#DEBUG  print(f'my_sentences datatype: {type(my_sentences)}')
#DEBUG  time.sleep(4)

for my_sentence in my_sentences:
    pprint(f'my_sentence datatype: {type(my_sentence)}', width = 140)
    pprint(my_sentence, width=4000)
    
#DEBUG sys.exit(0)

################
# Continuing Code
################

# Define sentences
# sentences = [  # Defining a list of example sentences
#     "I eat mango",
#     "mango is my favorite fruit",
#     "mango, apple, oranges are fruits",
#     "fruits are good for health",
# ]
sentences = my_sentences

# Tokenization and Vocabulary Creation
vocabulary = set()  # Initializing an empty set to store unique words
for sentence in sentences:  # Iterating over each sentence in the list
    tokens = sentence.lower().split()  # Tokenizing the sentence by splitting on whitespace and converting to lowercase
    vocabulary.update(tokens)  # Updating the set of vocabulary with unique tokens

# Assign unique indices to vocabulary words
word_to_index = {word: i for i, word in enumerate(vocabulary)}  # Creating a dictionary mapping words to unique indices
my_vec_cnt :int = 0
# Vectorization
sentence_vectors = {}  # Initializing an empty dictionary to store sentence vectors
for sentence in sentences:  # Iterating over each sentence in the list
    tokens = sentence.lower().split()  # Tokenizing the sentence by splitting on whitespace and converting to lowercase
    vector = np.zeros(len(vocabulary))  # Initializing a numpy array of zeros for the sentence vector
    for token in tokens:  # Iterating over each token in the sentence
        vector[word_to_index[token]] += 1  # Incrementing the count of the token in the vector
        my_vec_cnt +=1
        if my_vec_cnt < 2:
            pprint(f'token: {token} \\n | word_2_index: {word_to_index} {'\n'}| vector[word_2_index[token]] = {vector[word_to_index[token]]} {'\n'}', width=1000)
    sentence_vectors[sentence] = vector  # Storing the vector for the sentence in the dictionary
    #x = 
    #pprint((f'vector: {str(x)}'), width=140)
    #pprint([prettyfloat(n) for n in map(prettyfloat, vector) if n > float(0.00)], width=140)
# Store in VectorStore
for sentence, vector in sentence_vectors.items():  # Iterating over each sentence vector
    vector_store.add_vector(sentence, vector)  # Adding the sentence vector to the VectorStore
   
#######################################
# Similarity Search
#######################################
#query_sentence = "Mango is the best fruit"  # Defining a query sentence
query_sentence = "which is the best offer?"
query_vector = np.zeros(len(vocabulary))  # Initializing a numpy array of zeros for the query vector
query_tokens = query_sentence.lower().split()  # Tokenizing the query sentence and converting to lowercase
for token in query_tokens:  # Iterating over each token in the query sentence
    if token in word_to_index:  # Checking if the token is present in the vocabulary
        query_vector[word_to_index[token]] += 1  # Incrementing the count of the token in the query vector

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)  # Finding similar sentences

# Display similar sentences
print("Query Sentence:", query_sentence)  # Printing the query sentence
print("Similar Sentences:")  # Printing the header for similar sentences
for sentence, similarity in similar_sentences:  # Iterating over each similar sentence and its similarity score
    print(f"\nSimilarity = {similarity:.4f} | sentence: {sentence} ")  # Printing the similar sentence and its similarity score