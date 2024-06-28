from sklearn import datasets,neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA

dataset = datasets.load_diabetes()
data_train, data_test, target_train, target_test = train_test_split(dataset.data, dataset.target, train_size = 0.8,random_state=0)

print(dataset.data[:5])

scaler = preprocessing.MinMaxScaler()
scaler.fit(data_train)
data_scaled_train = scaler.transform(data_train)
data_scaled_test = scaler.transform(data_test)

pca = PCA(n_components=4)
pca.fit(data_scaled_train)
#print(pca.explained_variance_ratio_)

data_lower_dim_train = pca.transform(data_scaled_train)
data_lower_dim_test = pca.transform(data_scaled_test)


highestScore = 100000000

for i in range(1,50):
    model = neighbors.KNeighborsRegressor( n_neighbors = i )
    model.fit(data_lower_dim_train, target_train)
    result = model.predict(data_lower_dim_test)

    score = mean_squared_error(y_true=target_test, y_pred=result)
    if score < highestScore:
        highestScore = score
        bestK = i

print(score, bestK)

print("real value: ", target_test[:5])
print("predicted value: ", result[:5])
