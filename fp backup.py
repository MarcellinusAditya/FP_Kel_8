import numpy as np
import cv2


def normalitation(input):
    output = input/255
    return output

def grayscale(image):
    gray = np.zeros(image.shape[:2], dtype=np.uint8) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b) 
    return gray        



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
    
    assert image.shape[2] == 3, "Citra harus memiliki 3 channel (RGB)"
    
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape[:2]

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    
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


kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

image = cv2.imread('kentang.jpg')

grayscale_image=grayscale(image)

norm_image=normalitation(grayscale_image) #grayscale
norm_image_rgb=normalitation(image) #rgb

conv_image=conv2d(norm_image,kernel)
conv_image_rgb=conv2d_rgb(norm_image_rgb,kernel)

conv_image=relu(conv_image)
conv_image_rgb=relu(conv_image_rgb)


max_pooling_image=max_pooling(conv_image,size=2,stride=2)
max_pooling_image_2=max_pooling(conv_image_rgb,size=2,stride=2)



inputs = max_pooling_image_2.flatten()
weights = np.random.rand(inputs.size, 3) 
bias = np.random.rand(3)
print(bias)

dense_output = dense(inputs, weights, bias)

probabilities = softmax(dense_output)
print(dense_output)
print(probabilities)
# print(relu_image_2)

