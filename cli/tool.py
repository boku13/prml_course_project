# dropped the idea of a cli-tool, but just in case

# import argparse
# import matplotlib.pyplot as plt
# import numpy as np

# from data import cifar10


# def retrieve_similar_images(image_index):
#     # Placeholder logic for image retrieval
#     indices = np.random.choice(range(10000), 5, replace=False)
#     return indices

# def main():
#     parser = argparse.ArgumentParser(description='Image Retriever on the CIFAR-10 Dataset')
#     args = parser.parse_args()

#     print("Welcome to the CIFAR-10 Image Retriever!")
#     dataset = cifar10()

#     print("Displaying an image from each of the 10 classes:")
#     dataset.display_images(plt)

#     user_input = input("Enter the class number you're interested in (0-9): ")
#     selected_class = int(user_input)

#     print(f"Retrieving similar images for class: {dataset.classes[selected_class]}")
#     similar_indices = retrieve_similar_images(selected_class)
#     for idx in similar_indices:
#         print(f"Displaying similar image at index: {idx}")
#         # Placeholder for displaying images.

# if __name__ == "__main__":
#     main()
