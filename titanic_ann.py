import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import pandas as pd

from titanic import X_train, Y_train, X_test, Y_validate

batch_size = 6
epochs = 600

model = Sequential([
    keras.layers.Dense(
        units=53, 
        input_shape=(7,), 
        activation=tf.nn.relu
        ),
    keras.layers.Dense(
        units=27, 
        activation=tf.nn.relu, 
        kernel_regularizer=regularizers.l2(0.01)
        ),
    keras.layers.Dense(
        units=1, 
        activation=tf.nn.sigmoid
        )
])

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam'
)

model.fit(
    X_train.values, 
    Y_train.values,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
)

Y_pred = model.predict(X_test.values).tolist()
Y_pred = list(map(lambda x: 1 if x[0] > 0.5 else 0, Y_pred))

submission = pd.DataFrame({
    "PassengerId": Y_validate,
    "Survived": Y_pred
})

submission.to_csv(
    "submission_ann.csv", 
    index=False
)