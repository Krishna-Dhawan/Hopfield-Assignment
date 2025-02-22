import numpy as np # type: ignore
import os

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

if __name__ == "__main__":
    dataset = load_dataset("dataset")
    W = train(dataset)
    print("hello world")
