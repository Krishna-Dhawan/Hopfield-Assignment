import os
import random
import matplotlib.pyplot as plt


def read_pbm(filename):
    """
    :param filename: name of image in dataset
    :return: pixel value in a matrix
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = [[int(pixel) for pixel in line.split()] for line in lines[2:]]
    return data


def train_hopfield(images):
    """
    :param images: list containing all the images in the dataset
    :return: Weight matrix (16*16 x 16*16) of the trained hopfield network
    """
    num_pixels = 16 * 16
    W = [[0 for _ in range(num_pixels)] for _ in range(num_pixels)]

    for img in images:
        pattern = [(pixel * 2 - 1) for row in img for pixel in row]  # turn 1s and 0s into 1s and -1s
        for i in range(num_pixels):
            for j in range(num_pixels):
                if i != j:
                    # w[1, j] = (1/n) * sum[mu = 1 to n](E[mu, i] * E[mu, j])
                    # where E[mu, i] = bit i from pattern mu
                    W[i][j] += pattern[i] * pattern[j]

    return [[W[i][j] / len(images) for j in range(num_pixels)] for i in range(num_pixels)]


def corrupt_flip(images, prob):
    """
    :param images: list of all images
    :param prob: probability of flipping the bit
    :return: list of corrupted images
    """
    corrupted_images = []
    for img in images:
        # 1 - pixel flips the state
        corrupted_img = [[1 - pixel if random.random() < prob else pixel for pixel in row] for row in img]
        corrupted_images.append(corrupted_img)
    return corrupted_images


def corrupt_crop(images):
    """
    :param images: list of all images
    :return: image with all 0s except for a randomly selected 10x10 square
    """
    corrupted_images = []
    for img in images:
        x = random.randint(0, 5)
        y = random.randint(0, 5)
        corrupted_img = [[pixel if x <= i < x + 10 and y <= j < y + 10 else 0 for j, pixel in enumerate(row)] for i, row
                         in enumerate(img)]
        corrupted_images.append(corrupted_img)
    return corrupted_images


def correct_synchronous(image, W):
    """
    :param image, W: corrupted image, weight matrix of hopfield network
    :return: corrected image by synchronous updates each step
    """
    state = [(pixel * 2 - 1) for row in image for pixel in row]
    steps = 0
    num_pixels = 16 * 16

    while True:
        # make a new state at time t using values of states in time (t-1)
        new_state = [1 if sum(W[i][j] * state[j] for j in range(num_pixels)) >= 0 else -1 for i in range(num_pixels)]
        if new_state == state:  # network has converged
            break
        # change the state simultaneously: synchronous update
        state = new_state
        steps += 1
        if steps >= 100:
            break
    return [[(state[i * 16 + j] + 1) // 2 for j in range(16)] for i in range(16)], steps


def correct_asynchronous(image, W):
    """
    :param image, W: corrupted image, weight matrix of hopfield network
    :return: corrected image by asynchronous updates each step
    """
    state = [(pixel * 2 - 1) for row in image for pixel in row]
    steps = 0
    num_pixels = 16 * 16

    while True:
        # previous state stored for later comparison
        prev_state = state[:]
        for i in range(num_pixels):
            # change the states using current values: asynchronous updates
            state[i] = 1 if sum(W[i][j] * state[j] for j in range(num_pixels)) >= 0 else -1
        if state == prev_state:  # network has converged
            break
        steps += 1
        if steps >= 100:
            break
    return [[(state[i * 16 + j] + 1) // 2 for j in range(16)] for i in range(16)], steps


def compare_image(original, corrected):
    """
    :param original: original image from the dataset
    :param corrected: the one converged to by the Hopfield Network
    :return: bool: whether corrected is either exactly the same or inverted version of original
    """
    anti_corrected = [[1 - pixel for pixel in row] for row in corrected]  # Create anti-image
    return original == corrected or original == anti_corrected


def main():
    # create list of all images and train the network
    dataset = [read_pbm(f"./dataset/{filename}") for filename in sorted(os.listdir("./dataset"))]
    W = train_hopfield(dataset)

    probabilities = range(2, 10)
    convergence_rates_flip = []
    steps_histogram_data = {}

    # get data for successful converges when the pixels are flipped
    # with probability prob for each prob
    for prob in probabilities:
        successful_convergences = 0
        steps_data = []
        run = 0
        # 20 batches for each probability
        for _ in range(20):
            corrupted_images = corrupt_flip(dataset, prob * 0.1)
            run += 1
            imgn = 0
            for img, original in zip(corrupted_images, dataset):
                imgn += 1
                # try to correct the images first synchronously then asynchronously
                corrected_sync, steps_sync = correct_synchronous(img, W)
                corrected_async, steps_async = correct_asynchronous(img, W)
                if compare_image(original, corrected_sync):
                    successful_convergences += 1
                    steps_data.append(steps_sync)
                    print(
                        f"Probability : {prob * 0.1} Run {run} Image {imgn} successfully converges in {steps_sync} Synchronous steps")
                if compare_image(original, corrected_async):
                    successful_convergences += 1
                    steps_data.append(steps_async)
                    print(
                        f"Probability : {prob * 0.1} Run {run} Image {imgn} successfully converges in {steps_async} Asynchronous steps")

        convergence_rates_flip.append(successful_convergences / (40 * len(dataset)))
        steps_histogram_data[prob] = steps_data

    # get data for successful converges when the images are cropped
    # for each image
    successful_convergences = [0, 0, 0, 0]
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

    # plot data for converges when images are cropped
    plt.figure(figsize=(8, 5))
    plt.bar([img_ for img_ in ["circle", "cross", "plus", "square"]], successful_convergences, color='red')
    plt.xlabel("Correct Image")
    plt.ylabel("Converged Runs")
    plt.title("Convergence for cropped images vs Correct Image")
    plt.show()

    # plot data for converges when pixels are randomly flipped
    plt.figure(figsize=(8, 5))
    plt.bar([("0." + str(p)) for p in probabilities], convergence_rates_flip, color='blue')
    plt.xlabel("Flip Probability")
    plt.ylabel("Fraction of Correctly Converged Runs")
    plt.title("Convergence Rate vs Flip Probability")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar([("0." + str(p)) for p in probabilities], [1-val for val in convergence_rates_flip], color='blue')
    plt.xlabel("Flip Probability")
    plt.ylabel("Fraction of incorrectly/non-Converged Runs")
    plt.title("Rate vs Flip Probability")
    plt.show()

    # plot histograms for no. steps to convergence for each prob. of flip
    for prob, steps in steps_histogram_data.items():
        if steps:
            plt.figure(figsize=(8, 5))
            plt.hist(steps, bins=range(1, max(steps) + 2), color='green', edgecolor='black', alpha=0.7)
            plt.xlabel("Number of Steps to Convergence")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Steps to Convergence for p=0.{prob}")
            plt.show()
        else:
            print(f"No successful converges for p={prob * 0.1}")
    return 0


if __name__ == "__main__":
    main()
