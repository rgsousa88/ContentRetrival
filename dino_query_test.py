import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import faiss
import os
import argparse
from time import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DINOTest():
    def __init__(self, key_img_path:str):
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.model.eval()
        self.allow_file_ext = ('.jpg','.png')
        self.image_paths = [os.path.join(key_img_path, img) for img in os.listdir(key_img_path) if img.endswith(self.allow_file_ext)]
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std=[0.229, 0.224, 0.225])])
        start = time()
        self.build_faiss_index()
        print(f"Initialize time: {time() - start}s")

    def compute_l2_dist(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def compute_cossine_dist(self, vec1, vec2):
        vec1n = (1.0/np.linalg.norm(vec1)) * vec1
        vec2n = (1.0/np.linalg.norm(vec2)) * vec2
        distL2 = self.compute_l2_dist(vec1n, vec2n)
        return 1.0 - 0.5 * distL2 * distL2

    # Preprocess the image for DINO input
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        return img

    # Extract features using DINO model
    def extract_features(self, image_path):
        img = self.preprocess_image(image_path)
        with torch.no_grad():
            features = self.model(img).squeeze().numpy()
            nFeatures = (1.0/np.linalg.norm(features)) * features
        return nFeatures

    # Build a FAISS index for efficient similarity search
    def build_faiss_index(self,):
        features_list = []
        for path in self.image_paths:
            features = self.extract_features(path)
            features_list.append(features)
        
        self.features_array = np.array(features_list).astype('float32')
        print(f"Feat Arr {self.features_array.shape}")
        self.index = faiss.IndexFlatL2(self.features_array.shape[1])
        #self.index = faiss.IndexFlatIP(self.features_array.shape[1])
        self.index.add(self.features_array)

    # Perform content retrieval using FAISS
    def __retrieve_images(self, query_image_path:str, top_k=5):
        query_features = self.extract_features(query_image_path)
        query_features = np.array([query_features]).astype('float32')
        distances, indices = self.index.search(query_features, top_k)
        retrieved_paths = [self.image_paths[i] for i in indices[0]]
        #print(retrieved_paths)
        #print(distances[0])
        return retrieved_paths, distances[0]
    
    def retrieve_images(self, query_path:str):
        closest = []
        if os.path.isdir(query_path):
            files = [os.path.join(query_path, img) for img in os.listdir(query_path) if img.endswith(self.allow_file_ext)]
            for file in files:
                ret_path, dist = self.__retrieve_images(file, top_k=1)
                closest.append((ret_path[0], dist[0]))
        else:
            ret_paths, dists = self.__retrieve_images(query_path)
            for i, ret_path in enumerate(ret_paths):
                closest.append((ret_path, dists[i]))

        return closest


# Main function to experiment with DINO feature extraction and retrieval
def main(query_image_path: str):
    dataset_dir = 'data/key_images/'
    dino = DINOTest(dataset_dir)

    # Perform content retrieval with a query image
    retrieves = dino.retrieve_images(query_image_path)

    # Display the retrieved images and their distances
    print("Retrieved Images:")
    for i, item in enumerate(retrieves):
        path, distance = item
        print(f"{i+1}. {path} (Distance: {distance:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file path')
    args = parser.parse_args()
    print(f"Input file: {args.input}")

    main(args.input)
