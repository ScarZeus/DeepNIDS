import numpy as np
import pandas as pd
from conv1d import CNN1D
from conv1d_layers.sdg import SGD
from src.conv1d_layers import save_model

from conv1d_layers.soft_max_cross_entrophy import SoftmaxCrossEntropy

def main():
    
    data = pd.read_csv("data/df_minimization.csv")

    
    epochs = 5
    batch_size = 32
    learning_rate = 0.01
    num_classes = 2


    
    X_train = np.random.randn(data.shape[0], data.shape[1], 1)
    y_train = np.random.randint(0, num_classes, size=1024)

    X_val = np.random.randn(256, 70, 1)
    y_val = np.random.randint(0, num_classes, size=256)


    model =  CNN1D(num_classes=num_classes)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=learning_rate) 

   
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        total_loss = 0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            x_batch = X_train[start:end]
            y_batch = y_train[start:end]

    
            logits = model.forward(x_batch, training=True)
            loss = loss_fn.forward(logits, y_batch)
            total_loss += loss

            grad = loss_fn.backward()
            model.backward(grad)

            optimizer.step([
                model.block1.conv,
                model.block2.conv,
                model.fc.dense
            ])

        avg_loss = total_loss / num_batches

      
        correct = 0
        total = 0

        for i in range(0, len(X_val), batch_size):
            x = X_val[i:i+batch_size]
            y = y_val[i:i+batch_size]

            logits = model.forward(x, training=False)
            preds = np.argmax(logits, axis=1)

            correct += np.sum(preds == y)
            total += len(y)

        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print("-" * 40)


    save_model("cnn_cicids_from_scratch.npz", [
        model.block1.conv,
        model.block2.conv,
        model.fc.dense
    ])

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
