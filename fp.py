import numpy as np
import cv2
import os

def normalitation(input):
    output = input/255
    return output


def import_image(folder_path):
    images_array = []
    labels = []

   
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)  
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                images_array.append(img_rgb)  
                
                
                label = os.path.basename(root)  # Nama folder sebagai label
                if label == 'Early_Blight':
                    label = [1,0,0]
                elif label == 'Healthy':
                    label = [0,1,0]
                elif label == 'Late_Blight':
                    label = [0,0,1]
                labels.append(label)  
    
    return images_array,labels

def conv2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
    return output

def conv2d_rgb(image, kernel):
   
    assert image.shape[2] == 3
    
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape[:2]

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    # Proses setiap channel (R, G, B) secara independen
    for c in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                output[i, j] += np.sum(image[i:i+kernel_height, j:j+kernel_width, c] * kernel)

    return output


def max_pooling(feature_map, size=2, stride=2):
    pooled_height = (feature_map.shape[0] - size) // stride + 1
    pooled_width = (feature_map.shape[1] - size) // stride + 1

    pooled = np.zeros((pooled_height, pooled_width))

    for i in range(0, pooled_height, stride):
        for j in range(0, pooled_width, stride):
            pooled[i//stride, j//stride] = np.max(feature_map[i:i+size, j:j+size])

    return pooled

def relu(x):
    return np.maximum(0, x)

def dense(inputs, weights, bias):
    return relu(np.dot(inputs, weights) + bias)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Untuk stabilitas numerik
    return exp_x / np.sum(exp_x)


def forward_pass(image, kernel, weights, bias):
    image = normalitation(image)

    # Langkah 1: Konvolusi
    conv_output = conv2d_rgb(image, kernel)

    # Langkah 2: Max pooling
    pooled_output = max_pooling(conv_output)

    # Langkah 3: Flattening
    flattened_output = pooled_output.flatten()

    assert weights.shape[0] == flattened_output.size, f"Dimensi bobot {weights.shape} tidak cocok dengan flattening {flattened_output.size}"

    # Langkah 4: Fully connected layer
    dense_output = dense(flattened_output, weights, bias)

    # Langkah 5: Softmax
    probabilities = softmax(dense_output)
    
    return probabilities

def cross_entropy_loss(predictions, targets):
   
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(predictions))

def backward_pass(weights, bias, predictions, targets, flattened_output, learning_rate=0.01):
    # Gradien untuk lapisan output (dense layer)
    error = predictions - targets

    # Perbarui bobot dan bias
    weights_gradient = np.dot(flattened_output.reshape(-1, 1), error.reshape(1, -1))
    bias_gradient = error

    weights_gradient=weights_gradient[:16129, :]

    # Pembaruan bobot dengan gradient descent
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient
    
    return weights, bias

def train(images, labels, kernel, weights, bias, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        
        for i in range(len(images)):
            # Forward pass
            probabilities = forward_pass(images[i], kernel, weights, bias)

            # Hitung loss
            loss = cross_entropy_loss(probabilities, labels[i])
            total_loss += loss

            # Prediksi kelas dengan probabilitas tertinggi
            predicted_class = np.argmax(probabilities)
            actual_class = np.argmax(labels[i])

            if predicted_class == actual_class:
                print('Benar')
                correct_predictions += 1
            else:
                print('Salah')
            # Backward pass untuk memperbarui bobot
            weights, bias = backward_pass(weights, bias, probabilities, labels[i], images[i].flatten(), learning_rate)

        # Hitung akurasi
    
        accuracy = correct_predictions / len(images)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}, Accuracy: {accuracy*100}%")
    
    return weights, bias

def test(images, labels, kernel, weights, bias):
    correct_predictions = 0

    for i in range(len(images)):
        # Forward pass untuk test set
        probabilities = forward_pass(images[i], kernel, weights, bias)

        # Prediksi kelas
        predicted_class = np.argmax(probabilities)
        actual_class = np.argmax(labels[i])
        
        if predicted_class == actual_class:
            correct_predictions += 1

    # Hitung akurasi
    accuracy = correct_predictions / len(images)
    print(f"Test Accuracy: {accuracy*100}%")


# Inisialisasi kernel, bobot, dan bias secara acak
kernel_rgb = np.random.randn(3, 3)  
weights_rgb = np.random.randn(16129, 3)  
bias_rgb = np.random.randn(3)

# Data latihan 
x_train,y_train = import_image('Data Kentang/Training')
print(len(x_train))

# Latih model dengan dataset
weights_rgb, bias_rgb = train(x_train, y_train, kernel_rgb, weights_rgb, bias_rgb, epochs=10, learning_rate=0.01)

# Data uji
x_test,y_test=import_image('Data Kentang/Testing')


# Uji model setelah pelatihan
test(x_test, y_test, kernel_rgb, weights_rgb, bias_rgb)











# x_test = np.random.rand(20, 5, 5, 3)  # 20 gambar RGB untuk pengujian
# y_test = np.eye(3)[np.random.choice(3, 20)]



































# kernel = np.array([[1, 0, -1],
#                    [1, 0, -1],
#                    [1, 0, -1]])

# image = cv2.imread('kentang.jpg')

# grayscale_image=grayscale(image)

# norm_image=normalitation(grayscale_image)
# norm_image_rgb=normalitation(image)

# conv_image=conv2d(norm_image,kernel)
# conv_image_rgb=conv2d_rgb(norm_image_rgb,kernel)

# max_pooling_image=max_pooling(conv_image,size=2,stride=2)
# max_pooling_image_2=max_pooling(conv_image_rgb,size=2,stride=2)

# relu_image=relu(max_pooling_image)
# relu_image_2=relu(max_pooling_image_2)

# inputs = relu_image.flatten()
# weights = np.random.rand(inputs.size, 3) 
# bias = np.random.rand(3)

# dense_output = dense(inputs, weights, bias)

# probabilities = softmax(dense_output)
# print(dense_output)
# print(probabilities)
# print(relu_image_2)

# cv2.imshow('kentang', relu_image)
# cv2.imshow('kentang_2', relu_image_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()