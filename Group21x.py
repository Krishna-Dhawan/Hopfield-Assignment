import os
import random
import matplotlib.pyplot as plt

def read_pbm(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = [[int(pixel) for pixel in line.split()] for line in lines[2:]]
    return data

def write_pbm(filename, image):
    with open(filename, 'w') as f:
        f.write("P1\n16 16\n")
        for row in image:
            f.write(" ".join(map(str, row)) + "\n")

def train_hopfield(images):
    num_pixels = 16 * 16
    W = [[0 for _ in range(num_pixels)] for _ in range(num_pixels)]
    
    for img in images:
        pattern = [(pixel * 2 - 1) for row in img for pixel in row]
        for i in range(num_pixels):
            for j in range(num_pixels):
                if i != j:
                    W[i][j] += pattern[i] * pattern[j]
    
    return [[W[i][j] / len(images) for j in range(num_pixels)] for i in range(num_pixels)]

def corrupt_flip(images, prob):
    corrupted_images = []
    for img in images:
        corrupted_img = [[1 - pixel if random.random() < prob else pixel for pixel in row] for row in img]
        corrupted_images.append(corrupted_img)
    return corrupted_images

def corrupt_crop(images):
    corrupted_images = []
    for img in images:
        x = random.randint(0, 6)
        y = random.randint(0, 6)
        corrupted_img = [[pixel if x <= i < x+10 and y <= j < y+10 else 0 for j, pixel in enumerate(row)] for i, row in enumerate(img)]
        corrupted_images.append(corrupted_img)
    return corrupted_images


def correct_synchronous(image, W):
    state = [(pixel * 2 - 1) for row in image for pixel in row]
    steps = 0
    num_pixels = 16 * 16
    
    while True:
        new_state = [1 if sum(W[i][j] * state[j] for j in range(num_pixels)) >= 0 else -1 for i in range(num_pixels)]
        if new_state == state:
            break
        state = new_state
        steps += 1
        if steps >= 100:
            break
    return [[(state[i * 16 + j] + 1) // 2 for j in range(16)] for i in range(16)], steps

def correct_asynchronous(image, W):
    state = [(pixel * 2 - 1) for row in image for pixel in row]
    steps = 0
    num_pixels = 16 * 16
    
    while True:
        prev_state = state[:]
        for i in range(num_pixels):
            state[i] = 1 if sum(W[i][j] * state[j] for j in range(num_pixels)) >= 0 else -1
        if state == prev_state:
            break
        steps += 1
        if steps >= 100:
            break
    return [[(state[i * 16 + j] + 1) // 2 for j in range(16)] for i in range(16)], steps


def compare_image(original, corrected):
    anti_corrected = [[1 - pixel for pixel in row] for row in corrected]  # Create anti-image
    return original == corrected or original == anti_corrected


def main():
    dataset = [read_pbm(f"./dataset/{filename}") for filename in sorted(os.listdir("./dataset"))]
    W = train_hopfield(dataset)
    
    probabilities = range(2, 10)
    convergence_rates_flip = []
    steps_histogram_data = {}
    
    for prob in probabilities:
        successful_convergences = 0
        steps_data = []
        run = 0
        for _ in range(20):
            corrupted_images = corrupt_flip(dataset, prob*0.1)
            run += 1
            imgn = 0
            for img, original in zip(corrupted_images, dataset):
                imgn += 1 
                corrected_sync, steps_sync = correct_synchronous(img, W)
                corrected_async, steps_async = correct_asynchronous(img, W)
                if compare_image(original, corrected_sync):
                    successful_convergences += 1
                    steps_data.append(steps_sync)
                    print(f"Probability : {prob*0.1} Run {run} Image {imgn} successfully converges in {steps_sync} Synchronous steps")
                if compare_image(original, corrected_async):
                    successful_convergences += 1
                    steps_data.append(steps_async)
                    print(f"Probability : {prob*0.1} Run {run} Image {imgn} successfully converges in {steps_async} Asynchronous steps")
                            
        convergence_rates_flip.append(successful_convergences / (40 * len(dataset)))
        steps_histogram_data[prob] = steps_data

    successful_convergences = [0, 0, 0, 0]
    steps_data = []
    for _ in range(20):
        corrupted_images = corrupt_crop(dataset)
        for img, original in zip(corrupted_images, dataset):
            corrected_sync, steps_sync = correct_synchronous(img, W)
            corrected_async, steps_async = correct_asynchronous(img, W)
            for i in range(4):
                if corrected_sync[i] == original[i]:
                    successful_convergences[i] += 1
                if corrected_async[i] == original[i]:
                    successful_convergences[i] += 1
    
    plt.figure(figsize=(8, 5))
    plt.bar([img_ for img_ in ["circle", "cross", "plus", "square"]], successful_convergences, color='blue')
    plt.xlabel("Correct Image")
    plt.ylabel("Converged Runs")
    plt.title("Convergence for cropped images vs Correct Image")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar([str(p) for p in probabilities], convergence_rates_flip, color='blue')
    plt.xlabel("Flip Probability (*10)")
    plt.ylabel("Fraction of Converged Runs")
    plt.title("Convergence Rate vs Flip Probability")
    plt.show()
    
    for prob, steps in steps_histogram_data.items():
        if steps:
            plt.figure(figsize=(8, 5))
            plt.hist(steps, bins=range(1, max(steps)+2), color='green', edgecolor='black', alpha=0.7)
            plt.xlabel("Number of Steps to Convergence")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Steps to Convergence for p={prob}")
            plt.show()
        else:
            print(f"No successful converges for p={prob*0.1}")

if __name__ == "__main__":
    main()
