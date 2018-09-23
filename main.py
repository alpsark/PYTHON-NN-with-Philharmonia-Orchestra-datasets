import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import get_data


plt.interactive(False)

print('lets go')

#import datasets
decider= get_data.get_signals(violin=True , trupet=True ,flute = True , guitar = True , cello=False , viola = False)
viola_signals =  decider.read_viola()
print('viola finished with size ' , np.size(viola_signals))

flute_signals =  decider.read_flute()
print('flute finished with size ' , np.size(flute_signals))

violin_signals =  decider.read_violin()
print('violin finished with size ' , np.size(violin_signals))

trumpet_signals =  decider.read_trumpet()
print('trumpet finished with size ' , np.size(trumpet_signals))

guitar_signals =  decider.read_guitar()
print('guitar finished with size ' , np.size(guitar_signals))

cello_signals =  decider.read_cello()
print('cello finished with size ' , np.size(cello_signals))




#machine learning part
import Machine_learn
from sklearn import cross_validation
nn = Machine_learn.NN_1HL(hidden_layer_size=5)

#import sklearn.datasets as datasets
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

#file = open("X.txt", "w")
#for item in decider.X:
#   file.write("%s\n" % item)
#file.close()

#file = open("Y.txt", "w")
#file.write(decider.Y)
#file.close()

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(np.array(decider.X), np.array(decider.Y), test_size=0.25)

nn.fit(X_train, Y_train)



from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test, nn.predict(X_test))
print("accuracy: " , score )