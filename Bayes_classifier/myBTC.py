# librerías necesarias
import os
import nltk
import itertools
import numpy as np
import matplotlib.pyplot as plt
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Estableciendo stopwords y símbolos raros
bad_chars = '~:+[\@^{%(-"*”“|’,&<`}._=]!>;?#$)/'
stopwords.words('english')

# Función  para crear el mega documento
def MergeFiles(datapath):
# Hallando todos los documentos en el folder
    inputs = []
    for file in os.listdir(datapath):
        if file.endswith(".txt"):
            inputs.append(os.path.join(datapath, file))
# Concatenando todos los archivos en uno solo
    with open(datapath+'/merged_file.txt', 'w') as outfile:
        for fname in inputs:
            with open(fname, encoding="utf-8", errors='ignore') as infile:
                for line in infile:
                    outfile.write(line)

# Función que concatena los strings de una lista
def SumStr(x):
    result = ''
    for i in x:
        result = result + ' ' + i
    return result

# Función que pre-procesa los datos
def CleanData(file):
# Filtrando los símbolos raros
    filtered_bad_chars = ''.join(x for x in file.lower() if x not in bad_chars)
# Tokenizando
    text = word_tokenize(filtered_bad_chars)
# Retornando los toknes sin las stopwords
    return [word for word in text if not word in stopwords.words()]

# Función que crea la bolsa de palabras
def Bag_of_Words(document, k):
# Limpiando y tokenizando el documento
    tokenization = CleanData(document)
# Contando las veces que las palabras se repiten
    freqs = nltk.FreqDist(tokenization)
# Creando una lista con las palabras y su frecuencia
    freqs_list = [(k, v) for k, v in freqs.items()]
# Filtrando aquellas palabras con un umbral mayor a k
    filter_list =filter(lambda l: l[1] > k, freqs_list)
# Retornando la bolsa de palabras en orden descendente
    return sorted(filter_list,key=lambda m:m[1], reverse=True)

# Función que cálcula la probabilidades de un documento para pertener a una clase
def ProbabilitySet(document,document_class, total_classes, m):
# Bolsa de palabras de todas las clases
    T_string = Bag_of_Words(total_classes,m)
# Bolsa de palabras del documento desconocido 
    queryBag = Bag_of_Words(document,m)
# Bolsa de palabras de la clase a comparar
    classBag = Bag_of_Words(document_class,m)
# Filtrando sólo las palabras
    Tlist = [row[0] for row in T_string]
    query = [row[0] for row in queryBag]
    dclass= [row[0] for row in classBag]
    probal = []
# Probabilidad condicional con el suavizado de Laplace
    for i in query:
        feq = query.count(i)
        Cwc = dclass.count(i)
        c = len(dclass)
        T = Tlist
        p = (Cwc + 1)/(c + len(T))
        probal.append([i,feq,p])
# Retornando el documento con su frecuencia y probabilidad
    return probal

# Función auxiliar para hallar el producto de un conjunto 
def Aux(list):
    p = 10/50
    result = []
    for i in list:
        result.append(pow(i[2],i[1]) )
    
    return p*(np.prod(result))

# Función que clasifica el documento
def Classification(document,document_class, total_classes, m):
# Calculando las probabilidades de que el documento pertenezca a cada clase
    avehiconditional = ProbabilitySet(document,document_class[0], total_classes, m)
    amexiconditional = ProbabilitySet(document,document_class[1], total_classes, m)
    movieconditional = ProbabilitySet(document,document_class[2], total_classes, m)
    poetrconditional = ProbabilitySet(document,document_class[3], total_classes, m)
    storyconditional = ProbabilitySet(document,document_class[4], total_classes, m)
    Pvehic = Aux(avehiconditional)
    Pmexic = Aux(amexiconditional)
    Pmovie = Aux(movieconditional)
    Ppoetr = Aux(poetrconditional)
    Pstory = Aux(storyconditional)
# Calculando la clase con la probabilidad mayor
    Probabilitis = [Pvehic, Pmexic, Pmovie, Ppoetr, Pstory]
    max_number = max(Probabilitis)
# Imprimiendo la clasificación hecha a partir de un diccionario
    choices = {0: 'Autonomous vehicle paper', 1: 'American mexican history', 2: 'Movies review', 3: 'Poetry', 4: 'Short tale' }
    print(choices.get(Probabilitis.index(max_number), 'Error!'))

# Función que grafica la matriz de confusión
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
# Calculando la "accuracy"
    accuracy = np.trace(cm) / float(np.sum(cm))
# Configurando la visualización
    if cmap is None:
        cmap = plt.get_cmap('OrRd')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
# Opción que normaliza los valores y actualiza el gráfico 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy))
    plt.show()
