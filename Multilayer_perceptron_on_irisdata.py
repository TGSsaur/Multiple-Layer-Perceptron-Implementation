from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
##loading iris dataset
iris = load_iris()
# deviding iris dataset into train data and test data
data = train_test_split(iris.data, iris.target,test_size=0.2)
train_d, test_d, train_labels, test_labels = data

# scaling the data and passing train data to fit
scaler = StandardScaler()
scaler.fit(train_d)

# scaling the train data
train_d = scaler.transform(train_d)
test_d = scaler.transform(test_d)
neurons=[]
accuracy=[]

############### Training multilayer Model ###############################3
for i in range(8,14):
    
    mlp = MLPClassifier(hidden_layer_sizes=(i, 5), max_iter=1000)


    mlp.fit(train_d, train_labels)

    p_train = mlp.predict(train_d)

    p_test = mlp.predict(test_d)
    accu=accuracy_score(p_test, test_labels)
    print("Accuracy for {} Neurons:{}".format(i,accu))
    neurons.append(i)
    accuracy.append(accu)
    

plt.plot(neurons,accuracy)
plt.xlabel("Neurons")
plt.ylabel("Accuracy")
plt.show()
print(classification_report(p_test, test_labels))