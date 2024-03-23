import pickle

from model import ShallowNeuralNetwork, ShallowNeuralNetworkConfig
from dataset import generate_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


def train_test_split_data(df, test_size=0.2):
    X = df[['Feature1', 'Feature2']].values
    Y = df['Label'].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42)

    return X_train.T, X_test.T, Y_train.reshape(1, -1), Y_test.reshape(1, -1)


def train():
    epochs = 100
    df = generate_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split_data(df)

    config = ShallowNeuralNetworkConfig()
    model = ShallowNeuralNetwork(config)

    for epoch in range(epochs):
        # Training
        A2_train = model.forward(X_train)
        loss = model.compute_loss(A2_train, Y_train)
        dW1, db1, dW2, db2 = model.backward(X_train, Y_train)
        model.update_parameters(dW1, db1, dW2, db2)

        # Validation
        A2_test = model.forward(X_test)
        val_loss = model.compute_loss(A2_test, Y_test)

        print(
            f'Epoch {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}\n')

        # Testing (every few epochs)
        if epoch % 10 == 0:
            predictions = np.argmax(A2_test, axis=0)
            y_test_labels = np.argmax(Y_test, axis=0)
            f1 = f1_score(y_test_labels, predictions, average='macro')
            print(f'Epoch {epoch}, F1 Score (Test): {f1}\n')

        # Save the best model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    train()
