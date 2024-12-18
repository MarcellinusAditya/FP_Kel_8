from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def normalitation(input):
    return input / 255.0

def import_image(folder_path):
    images_array = []
    labels = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (128, 128))
                images_array.append(normalitation(img_resized))

                # Menggunakan nama folder sebagai label
                label = os.path.basename(root)
                labels.append(label)

    return np.array(images_array), np.array(labels)

def encode_labels(labels):
    label_to_one_hot = {
        "Early_Blight": [1, 0, 0],
        "Healthy": [0, 1, 0],
        "Late_Blight": [0, 0, 1]
        }
    labels_one_hot = np.array([label_to_one_hot[label] for label in labels])
    
    return labels_one_hot


def conv2d(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def max_pooling(feature_map, size=2, stride=2):
    pooled_height = (feature_map.shape[0] - size) // stride + 1
    pooled_width = (feature_map.shape[1] - size) // stride + 1
    pooled = np.maximum.reduceat(
        np.maximum.reduceat(feature_map, np.arange(0, feature_map.shape[0], stride), axis=0),
        np.arange(0, feature_map.shape[1], stride), axis=1
    )
    return pooled[:pooled_height, :pooled_width]

def relu(x):
    return np.maximum(0, x)

def dense(inputs, weights, bias):
    return relu(np.dot(inputs, weights) + bias)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def forward_pass(image, kernel, weights, bias):
    
    conv_output = conv2d(image, kernel)
    pooled_output = max_pooling(conv_output)
    conv_output = conv2d(pooled_output, kernel)
    pooled_output = max_pooling(conv_output)
    flattened_output = pooled_output.flatten()
    dense_output = dense(flattened_output, weights, bias)
    return softmax(dense_output), flattened_output

def cross_entropy_loss(predictions, targets):
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(predictions))

def backward_pass(weights, bias, predictions, targets, flattened_output, learning_rate=0.01):
    error = predictions - targets
    weights_gradient = np.outer(flattened_output, error)
    bias_gradient = error
    assert weights_gradient.shape == weights.shape, "Gradient dimensions do not match weights"
    # weights_gradient=weights_gradient[:768, :]
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient
    return weights, bias

def train(images, labels,x_test, y_test, kernel, weights, bias, epochs=10, learning_rate=0.01):
    epoch_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        for i in range(len(images)):
            
            # print(f"Flattened input size: {flattened_size}")
            # print(f"Weights shape: {weights.shape}")
            probabilities, flattened_output= forward_pass(images[i], kernel, weights, bias)
            loss = cross_entropy_loss(probabilities, labels[i])
            total_loss += loss
            predicted_class = np.argmax(probabilities)
            actual_class = np.argmax(labels[i])

            # if predicted_class == actual_class:
            #     print('Benar')
            #     correct_predictions += 1
            # else:
            #     print('Salah')

            if predicted_class == actual_class:
                correct_predictions += 1

            weights, bias = backward_pass(weights, bias, probabilities, labels[i], flattened_output, learning_rate)

        accuracy = correct_predictions / len(images)
        epoch_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}, Accuracy: {accuracy*100}%", end=' ')
        test_accuracies.append(test(x_test, y_test, kernel, weights, bias))

    visualize_accuracy(epoch_accuracies, test_accuracies)
    return weights, bias

def test(images, labels, kernel, weights, bias):
    correct_predictions = 0

    for i in range(len(images)):
        probabilities, flat = forward_pass(images[i], kernel, weights, bias)
        predicted_class= np.argmax(probabilities)
        actual_class = np.argmax(labels[i])

        if predicted_class == actual_class:
            correct_predictions += 1

    accuracy = correct_predictions / len(images)
    print(f"Test Accuracy: {accuracy*100}%")
    return accuracy

def visualize_accuracy(train_accuracies, test_accuracies):
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train & Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
# Inisialisasi kernel, bobot, dan bias secara acak
# kernel_rgb = np.random.normal(0, 0.1, (3, 3))
# weights_rgb = np.random.normal(0, 0.1,(3072, 3)) 
kernel_rgb = np.random.randn(3, 3) 
weights_rgb = np.random.randn(3072, 3) 
bias_rgb = np.random.randn(3)
epochs=20
folder_path = "Data Kentang All"

images, labels = import_image(folder_path)
print("Jumlah data:", len(images))


x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=0, stratify=labels
)

unique_labels, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label '{label}': {count} data")
unique_labels, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label '{label}': {count} data")
    
y_train=encode_labels(y_train)
y_test=encode_labels(y_test)


print("Jumlah data training:", len(x_train))
print("Jumlah data testing:", len(x_test))

# Latih model dengan dataset
weights_rgb, bias_rgb = train(x_train, y_train,x_test, y_test, kernel_rgb, weights_rgb, bias_rgb, epochs, learning_rate=0.001)


# Uji model setelah pelatihan
# test(x_test, y_test, kernel_rgb, weights_rgb, bias_rgb)














# if label == 'Early_Blight':
#                     label = [1, 0, 0]
#                 elif label == 'Healthy':
#                     label = [0, 1, 0]
#                 elif label == 'Late_Blight':
#                     label = [0, 0, 1]