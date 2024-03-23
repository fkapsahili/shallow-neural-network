import numpy as np
import pandas as pd

np.random.seed(42)


def generate_dataset(n_samples=10_000):
    mean_values = {
        0: [1, 2],
        1: [5, 7],
        2: [8, 1]
    }

    # Generate data
    features = []
    labels = []
    for class_label, mean in mean_values.items():
        class_features = np.random.randn(n_samples, 2) + mean
        class_labels = [class_label] * n_samples
        features.extend(class_features)
        labels.extend(class_labels)

    df = pd.DataFrame(features, columns=['Feature1', 'Feature2'])
    df['Label'] = labels

    return df
