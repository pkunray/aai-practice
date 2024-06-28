from sklearn import datasets,neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA

highestScore = 0
bestK = 0
#dataset = datasets.load_iris()

dataset = datasets.load_wine()
data_train, data_test, target_train, target_test = train_test_split(dataset.data, dataset.target, train_size = 0.8,random_state=10)

print(dataset.data[0])
#print(dataset.target)

scaler = preprocessing.MinMaxScaler()
scaler.fit(data_train)
data_scaled_train = scaler.transform(data_train)
data_scaled_test = scaler.transform(data_test)
print(data_scaled_train[0])

pca = PCA(n_components=5)
pca.fit(data_scaled_train)
print(pca.explained_variance_ratio_)

data_lower_dim_train = pca.transform(data_scaled_train)
data_lower_dim_test = pca.transform(data_scaled_test)

for i in range(1,50):
    model = neighbors.KNeighborsClassifier( n_neighbors = i )
    model.fit(data_lower_dim_train, target_train)

    result = model.predict(data_lower_dim_test)
    #print(result)
    #print(target_test[:2])

    score = accuracy_score(target_test, result)
    if score > highestScore:
        highestScore = score
        bestK = i
        #print(score)

print(highestScore, bestK)

confusion_matrix = confusion_matrix(y_true=target_test, y_pred=result)
print(confusion_matrix)