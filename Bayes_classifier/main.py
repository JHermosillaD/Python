import numpy as np
import myBTC

# Carpeta con los conjuntos de entrenamiento
vehicles = 'Data/Autonomous vehicle/Training'
mexico_h = 'Data/Mexican history/Training'
movies_r = 'Data/Movie review/Training'
poetry   = 'Data/Poetry/Training'
Stories  = 'Data/Short Storys/Training'

# Creando mega archivo
myBTC.MergeFiles(vehicles)
myBTC.MergeFiles(mexico_h)
myBTC.MergeFiles(movies_r)
myBTC.MergeFiles(poetry)
myBTC.MergeFiles(Stories)

# Definiendo la ruta del mega documento de entrenamiento para cada clase
vehicles_str = open('Data/Autonomous vehicle/Training/merged_file.txt').read()
mexico_h_str = open('Data/Mexican history/Training/merged_file.txt').read()
movies_r_str = open('Data/Movie review/Training/merged_file.txt').read()
poetry_str   = open('Data/Poetry/Training/merged_file.txt').read()
Stories_str  = open('Data/Short Storys/Training/merged_file.txt').read()

# Conjutno de entrenamiento total
TrainingSet = [vehicles_str, mexico_h_str, movies_r_str, poetry_str, Stories_str]

# Definiendo la ruta del conjunto de prueba de cada clase

# Definiendo la ruta de la clase 1
vehicles_d1 = open('Data/Autonomous vehicle/Tests/aerealvehicleleaders.txt').read()
vehicles_d2 = open('Data/Autonomous vehicle/Tests/ATRA.txt').read()
vehicles_d3 = open('Data/Autonomous vehicle/Tests/hexapod.txt').read()
vehicles_d4 = open('Data/Autonomous vehicle/Tests/quadrotor.txt').read()
vehicles_d5 = open('Data/Autonomous vehicle/Tests/radar.txt').read()

# Definiendo la ruta de la clase 2
mexico_h_d1 = open('Data/Mexican history/Tests/afromexico.txt').read()
mexico_h_d2 = open('Data/Mexican history/Tests/Biogeographical.txt').read()
mexico_h_d3 = open('Data/Mexican history/Tests/Population.txt').read()
mexico_h_d4 = open('Data/Mexican history/Tests/ruralhistory.txt').read()
mexico_h_d5 = open('Data/Mexican history/Tests/secondary.txt').read()

# Definiendo la ruta de la clase 3
movies_r_d1 = open('Data/Movie review/Test/Outside_The_Wire.txt').read()
movies_r_d2 = open('Data/Movie review/Test/Parasite.txt').read()
movies_r_d3 = open('Data/Movie review/Test/ShangChi.txt').read()
movies_r_d4 = open('Data/Movie review/Test/SnydersJusticeLeague.txt').read()
movies_r_d5 = open('Data/Movie review/Test/TheConjuring3.txt').read()

# Definiendo la ruta de la clase 4
poetry_d1 = open('Data/Poetry/Tests/allearth.txt').read()
poetry_d2 = open('Data/Poetry/Tests/AURORA.txt').read()
poetry_d3 = open('Data/Poetry/Tests/spring.txt').read()
poetry_d4 = open('Data/Poetry/Tests/testpoem.txt').read()
poetry_d5 = open('Data/Poetry/Tests/TheConscienceofAvimaelGuzman.txt').read()

# Definiendo la ruta de la clase 5
Stories_d1 = open('Data/Short Storys/Tests/TheAgedMother.txt').read()
Stories_d2 = open('Data/Short Storys/Tests/TheEyesHaveIt.txt').read()
Stories_d3 = open('Data/Short Storys/Tests/TheFableofthePreacher.txt').read()
Stories_d4 = open('Data/Short Storys/Tests/TheKingandHisHawk.txt').read()
Stories_d5 = open('Data/Short Storys/Tests/TheLittleMatchGirl.txt').read()


print('Vehicle test classification')
myBTC.Classification(vehicles_d1, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(vehicles_d2, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(vehicles_d3, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(vehicles_d4, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(vehicles_d5, TrainingSet,myBTC.SumStr(TrainingSet),2)

print('Mexican stories classification')
myBTC.Classification(mexico_h_d1, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(mexico_h_d2, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(mexico_h_d3, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(mexico_h_d4, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(mexico_h_d5, TrainingSet,myBTC.SumStr(TrainingSet),2)

print('Movies review classification')
myBTC.Classification(movies_r_d1, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(movies_r_d2, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(movies_r_d3, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(movies_r_d4, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(movies_r_d5, TrainingSet,myBTC.SumStr(TrainingSet),2)

print('Poems classification')
myBTC.Classification(poetry_d1, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(poetry_d2, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(poetry_d3, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(poetry_d4, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(poetry_d5, TrainingSet,myBTC.SumStr(TrainingSet),2)

print('Short stories classification')
myBTC.Classification(Stories_d1, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(Stories_d2, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(Stories_d3, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(Stories_d4, TrainingSet,myBTC.SumStr(TrainingSet),2)
myBTC.Classification(Stories_d5, TrainingSet,myBTC.SumStr(TrainingSet),2)

myBTC.plot_confusion_matrix(cm           = np.array([[ 5, 0, 0, 0, 0],
                                              [  0, 3, 0, 1, 1],
                                              [  0, 0, 4, 0, 1],
                                              [  0, 0, 0, 3, 2],
                                              [  0, 0, 1, 1, 3]]), 
                      normalize    = False,
                      target_names = ['Autonomous vehicles', 'Mexican american', 'Movies reviews', 'Long poem', 'Short stories'],
                      title        = "Confusion Matrix")
