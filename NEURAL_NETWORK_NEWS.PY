# NEURAL_NETWORK_NEWS.PY BY EMILIO MARTÍNEZ BARAJAS 1801473

import pandas as pd #IMPORTANDO LIBRERIAS
import os
from pandas import Series
import numpy as numpy
import matplotlib.pyplot as pyplot

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import csv
######################################################## MENU ########################################################
print("EMILIO MARTÍNEZ BARAJAS 1801784\n PROYECTO INTEGRADOR DE REDES NEURONALES \n\n\n ===MENU DE OPCIONES=== \n 0.-INSTRUCCIONES \n 1.-GRAFICAS DE LA BASE DE DATOS \n 2.-ENTRENAR LA RED NEURONAL \n 3.-REALIZAR UNA PREDICCIÓN \n\n\n 4.-Salir\n\n\n\n")
opcionMenu = input("Inserta un numero valor >> ")

if opcionMenu=="0":
	print("La base de datos real se encuentra en: haberman.csv \n\nEn caso de que se quieran introducir datos para su predicción se debe usar: Nuevos.csv\ny los resultados apareceran en: predict.csv \n\n")
	print("La base de datos tiene como objetivo almacenar los resultados de personas que sobrevivieron a una operación de\nextraccion de cáncer de mama entre los años 1958 y 1969.\n")
	print("Ésta cuenta con 4 columnas:\n   Edad del paciente.\n   Año de la operación.\n   Número de tumores adicionales encontrados. \n     1.-El paciente sobrevivió mas de 5 años. \n     2.-El paciente murió antes de 5 años.")
######################################################## ANALISIS Y GRAFICADO DE DATOS ########################################################
elif opcionMenu=="1":
	fig, axes = pyplot.subplots(nrows=2, ncols=2) # Se desean mostrar 4 gráficas 2x2=4



	data=pd.read_csv('haberman.csv', encoding = "ISO-8859-1") #Abriendo el archivo .csv y cambiando el codificado para que no de problemas

																			######## GRAFICA DE SOBREVIVENCIA ########
	dfsurvival= pd.DataFrame(data, columns=['SURVIVAL']) #Leer la columna survival
	count = dfsurvival.pivot_table(index=['SURVIVAL'], aggfunc='size') #realizar una lista con las veces que se repiten los datos
	ax=count.plot.bar(x='SURVIVAL', y='count', ax=axes[0,0]) #crear una grafica con x , y, ponerla en la casilla 0,0  del conjunto de graficas

	dfAge = pd.DataFrame(data, columns=['AGE'])								 ######## GRAFICA DE EDAD DE PACIENTES ########
	countAge = dfAge.pivot_table(index=['AGE'], aggfunc='size')
	axAge=countAge.plot(x='dfAge', y='countAge', ax=axes[0,1])

	dfOPERATION = pd.DataFrame(data, columns=['OPERATION'])                  ######## AÑO EN QUE SE REALIZO LA OPERACIÓN ########
	countOPERATION = dfOPERATION.pivot_table(index=['OPERATION'], aggfunc='size')
	axOPERATION=countOPERATION.plot(x='dfOPERATION', y='countOPERATION', ax=axes[1,0])

	dfNodes = pd.DataFrame(data, columns=['NODES'])                          ######## NUMERO DE TUMORES ADICIONALES ENCONTRADOS ########
	countNodes = dfNodes.pivot_table(index=['NODES'], aggfunc='size')
	axNodes=countNodes.plot(x='dfNodes', y='countNodes', ax=axes[1,1])

	pyplot.show() #mostrar graficas

################################################################### RED NEURONAL ###################################################################
elif opcionMenu=="2":
	X=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(0,1,2), skiprows=1)
	Y=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(3), skiprows=1, dtype='str')

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.38, random_state=3)

	clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(4, ), random_state=1, verbose=True)
	clf.fit(X_train, Y_train) 

	predicted = clf.predict(X_test)
	error= 1-accuracy_score(Y_test, predicted)

	print('\nError en el conjunto de prueba')
	print(error)

	filename = 'model.sav'
	pickle.dump(clf, open(filename, 'wb'))

################################################################### PREDICCION ###################################################################
elif opcionMenu=="3":
	
	loaded_model = pickle.load(open('model.sav', 'rb'))

	X=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(0,1,2), skiprows=1)
	Y=numpy.loadtxt('haberman.csv', delimiter=',', usecols=(3), skiprows=1, dtype='str')
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.38, random_state=3)

	
	X_test = numpy.loadtxt('Nuevos.csv', delimiter=',', usecols=(0,1,2), skiprows=1)



	DecisionTreeClassifierModel = DecisionTreeClassifier()
	DecisionTreeClassifierModel.fit(X_train, Y_train)
	Y_Pred = DecisionTreeClassifierModel.predict(X_test)
	print("\n Los datos pronosticados son:")
	print(Y_Pred)
	print("Los cuales se almacenan en predict.csv")

	f = open('predict.csv', 'w',newline='')

	with f:
	    writer = csv.writer(f)
	    writer.writerows(Y_Pred)

	predicted = loaded_model.predict(X_test)

	Y_test = numpy.loadtxt('predict.csv', encoding='ISO-8859-1', delimiter=',', usecols=(0), dtype='str')
	error= 1-accuracy_score(Y_test, predicted)

	print('\nError esperado en la prueba:')
	print(error)


################################################################### MAS INFO ###################################################################
else:
	DoNothing=0