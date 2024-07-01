import keras
from sklearn.datasets import load_wine

dataset = load_wine()

# Standardize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(dataset.data).data
y = dataset.target
print(X.shape)
# Get the unique classes
print(set(y))

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
#print(pca.explained_variance_)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(15, input_shape=(6,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[callback])

from matplotlib import pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

print(model.predict(X_test[:3]))
print(y_test[:3])