import numpy as np
import random;
import os
import matplotlib.pyplot as plt

def readpbm(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    width, height = 16, 16
    data = np.array([int(pixel) for line in lines[2:] for pixel in line.split()])
    return data.reshape((16, 16))

def write_pbm(filename, image):
    with open(filename, 'w') as file:
        file.write("P1\n16 16\n")
        for row in image:
            file.write(row + "\n")

def train(images):
    n = 16*16
    W = np.zeros((n, n))

    for img in images:
        pattern = img.flatten() * 2 - 1
        W += np.outer(pattern, pattern)
    np.fill_diagonal(W, 0)
    return W/len(images)

def load_dataset(dir):
    images = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".pbm"):
            images.append(readpbm(os.path.join(dir, filename)))
    return images

def corrupt_flip(images, prob):
    corrupted_images = []
    for img in images:
        for pixel in img.flatten():
            if (random.random() <= prob):
                pixel = 1-pixel
    return corrupted_images

def corrupt_crop(images):
    corrupted_images = []
    for img in images:
        corrupted_img = np.zeros_like(img)
        corrupted_img[3:13, 3:13] = img[3:13, 3:13]
        corrupted_images.append(corrupted_img)
    return corrupted_images

def correct_synchronous(images, W):
    steps = 0
    for img in images:
        state = img.flatten() * 2 - 1
        prev_state = np.zeros_like(state)
        while not np.array_equal(state, prev_state):
            prev_state = state.copy()
            state = np.sign(W @ state)
            state[state == 0] = 1
            steps += 1
    return steps

def correct_asynchronous(images, W):
    steps = 0
    for img in images:
        state = img.flatten() * 2 - 1
        prev_state = np.zeros_like(state)
        while not np.array_equal(state, prev_state):
            prev_state = state.copy()
            for i in np.random.permutation(len(state)):
                state[i] = np.sign(W[i] @ state)
                if state[i] == 0:
                    state[i] = 1
            steps += 1
    return steps

def compare(original_images, corrected_images):
    """Compares original images with corrected images and returns accuracy."""
    matches = sum(np.array_equal(orig, corr) for orig, corr in zip(original_images, corrected_images))
    return matches / len(original_images)


def main():
    dataset = load_dataset("dataset")
    W = train(dataset)
    corrupted_images = corrupt_flip(dataset, 0.3)
    corrected_sync = [correct_synchronous([img], W) for img in corrupted_images]
    corrected_async = [correct_asynchronous([img], W) for img in corrupted_images]
    
    accuracy_sync = compare(dataset, corrected_sync)
    accuracy_async = compare(dataset, corrected_async)
    
    print(f"Synchronous Correction Accuracy: {accuracy_sync:.2%}")
    print(f"Asynchronous Correction Accuracy: {accuracy_async:.2%}")
    
    # Plot example of original, corrupted, and corrected images
    fig, axes = plt.subplots(3, len(dataset), figsize=(12, 4))
    for i, (orig, corr, fixed) in enumerate(zip(dataset, corrupted_images, corrected_sync)):
        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].set_title("Original")
        axes[1, i].imshow(corr, cmap="gray")
        axes[1, i].set_title("Corrupted")
        axes[2, i].imshow(fixed, cmap="gray")
        axes[2, i].set_title("Corrected")
    plt.show()


if __name__ == "__main__":
    main()
