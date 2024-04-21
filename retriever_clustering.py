# retriever code updated to handle clustering models 
import pickle
import os
import numpy as np
from data import cifar10

class simple_retriever():
    def __init__(self, model, preprocessing, is_clustering=False):
        self.model = model
        self.transform = preprocessing
        self.model_path = "models/default/" + model + ".pkl"
        self.feature_transform_path = "models/default/" + preprocessing + ".pkl"
        self.is_clustering = is_clustering
        self.image = None
        self.loaded_model = None
        self.loaded_transform = None
        self.cluster_assignments = None  # Initialize cluster_assignments attribute
        
        # Initialize cluster assignments if it's a clustering model
        if self.is_clustering:
            self.init_cluster_assignments()

    def load_preprocessing(self):
        print(self.feature_transform_path)
        if os.path.exists(self.feature_transform_path):
            with open(self.feature_transform_path, 'rb') as file:
                self.loaded_transform = pickle.load(file)
        else:
            print("Preprocessing file does not exist.")

    def load_model(self):
        print(self.model_path)
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as file:
                self.loaded_model = pickle.load(file)
        else:
            print("Model file does not exist.")

    def init_cluster_assignments(self):
        data = cifar10()
        images = data.images  # Assuming images are already preprocessed if necessary
        if self.loaded_model is not None:
            # Initialize cluster assignments using the loaded model
            self.cluster_assignments = self.loaded_model.predict(images)
        else:
            print("Model is not loaded. Cannot initialize cluster assignments.")

    def inference(self, image):
        self.image = image
        if self.loaded_transform is not None:
            image = self.loaded_transform.transform([image])
            print(image.shape)
        if self.loaded_model is not None:
            if self.is_clustering:
                # For clustering models, return the cluster label
                label = self.loaded_model.predict(image)
                return label[0]
            else:
                # For classification models, return the class label
                label = self.loaded_model.predict([image])
                return label[0]
        else:
            return None

    def retrieve_images(self, label):
        data = cifar10()
        if self.is_clustering:
            # For clustering models, retrieve images belonging to the same cluster
            if self.cluster_assignments is not None:
                cluster_indices = np.where(self.cluster_assignments == label)[0]
                cluster_images = [data.images[i] for i in cluster_indices]
                return cluster_images[:10]  # Return only the first 10 images
            else:
                print("Cluster assignments are not initialized.")
                return []
        else:
            # For classification models, retrieve images based on class label
            if label in data.classes:
                # Use the label to retrieve images from the images_per_class dictionary
                images = data.images_per_class[label]
                return images
            else:
                print(f"Label '{label}' not found in the dataset.")
                return []
