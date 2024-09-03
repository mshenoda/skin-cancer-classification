from ultralytics import YOLO
import numpy as np
import cv2
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score
from utils.preprocess import clean_image
from utils.image_processing import zoom_in, histogram_stretch, apply_clahe

class SkinCancerClassifier:
    """
    A class for performing image classification using the YOLO model.

    Example Usage:
        import cv2
        from classifier import SkinCancerClassifier
        from utils.plotting import plot_images

        image = cv2.imread('path/to/image.jpg')

        classifier = SkinCancerClassifier(model_path='path/to/model.pt')

        class_summary, image_cleaned, hairline_mask, border_mask = classifier.predict(image)

        class_name = class_summary['name']
        class_conf = class_summary['confidence']

        plot_images([image, image_cleaned, hairline_mask, border_mask], 
                    [f"Input \n{image_path}", f"Classification \n'{class_name}' {class_conf:.1%}", "Hairline Mask", "Border Mask"])
    """
    def __init__(self, model_path):
        """
        Initializes the Classifier with a YOLO model for classification.

        Args:
            model_path (str): Path to the pre-trained YOLO model.
        """
        self._model = YOLO(model_path, task="classify", verbose=False)

    def _preprocess(self, image):
        """
        Preprocesses the input image by cleaning it and generating masks.
        
        Args:
            image (numpy.ndarray): The input image to preprocess.
        
        Returns:
            tuple: A tuple containing:
                - image_cleaned (numpy.ndarray): The cleaned image.
                - hairline_mask (numpy.ndarray): The mask for hairline regions.
                - border_mask (numpy.ndarray): The mask for border regions.
        """
        image_cleaned, hairline_mask, border_mask = clean_image(image)
        return image_cleaned, hairline_mask, border_mask

    def _inference(self, image):
        """
        Performs inference on the given image using the YOLO model.
        
        Args:
            image (numpy.ndarray): The image on which inference is to be performed.
        
        Returns:
            dict: A dictionary containing the classification summary with keys:
                - "class" (int): The predicted class ID.
                - "name" (str): The name of the predicted class.
                - "confidence" (float): The confidence score of the prediction.
        """
        results = self._model.predict(image, verbose=False)
        class_summary = results[0].summary(normalize=False, decimals=3)[0]
        return class_summary

    def _postprocess(self, image, image_cleaned, class_summary):
        """
        Post-processes the results based on the confidence of the initial classification.

        Args:
            image (np.ndarray): Original input image.
            image_cleaned (np.ndarray): Preprocessed cleaned image.
            class_summary (dict): Initial classification summary.

        Returns:
            dict: Final classification result after applying transformations and majority voting.
        """
        # If confidence is above 0.7, return the results directly
        if class_summary["confidence"] >= 0.7:
            return class_summary

        # Perform additional inferences with different transformations
        predictions = []

        # Original image
        predictions.append(self._inference(image))

        # Flipped horizontally
        image_flipped_horizontally = cv2.flip(image_cleaned, 1)
        predictions.append(self._inference(image_flipped_horizontally))

        # Flipped vertically
        image_flipped_vertically = cv2.flip(image_cleaned, 0)
        predictions.append(self._inference(image_flipped_vertically))

        # Enhanced image Applying CLAHE
        image_enhanced = apply_clahe(image_flipped_vertically)
        predictions.append(self._inference(image_enhanced))
        
        # Enhanced image Histogram Stretch
        image_enhanced = histogram_stretch(image_flipped_horizontally)
        predictions.append(self._inference(image_enhanced))
        
        # Zoomed in
        image_zoomed = zoom_in(image_cleaned, zoom_factor=0.07)
        predictions.append(self._inference(image_zoomed))

        # Perform majority voting based on the predictions
        final_prediction = self._majority_vote(predictions)
        return final_prediction

    def _majority_vote(self, predictions):
        """
        Performs majority voting to determine the final prediction from multiple inferences.

        Args:
            predictions (list): List of classification results from different transformations.

        Returns:
            dict: The final classification result based on majority voting.
        """
        class_ids = [pred["class"] for pred in predictions]
        confidences = [pred["confidence"] for pred in predictions]
        names = [pred["name"] for pred in predictions]

        # Find the most common class ID among the predictions
        most_common_class_id = max(set(class_ids), key=class_ids.count)

        # Filter predictions to only those matching the most common class ID
        matching_confidences = [confidences[i] for i in range(len(class_ids)) if class_ids[i] == most_common_class_id]
        matching_names = [names[i] for i in range(len(class_ids)) if class_ids[i] == most_common_class_id]

        # The name should correspond to the most common class ID
        most_common_name = matching_names[0]  # Assuming all names for this class ID are the same

        # Average the confidence of the most common class ID
        avg_confidence = round(np.mean(matching_confidences),3)

        return {"class": most_common_class_id, "name": most_common_name, "confidence": avg_confidence}

    def predict(self, image):
        """
        Predicts the class of the input image by performing preprocessing, 
        inference, and postprocessing steps.
        
        Args:
            image (numpy.ndarray): The input image to classify.
        
        Returns:
            tuple: A tuple containing:
                - class_summary (dict): The final classification summary.
                - image_cleaned (numpy.ndarray): The cleaned image used for inference.
                - hairline_mask (numpy.ndarray): The mask for hairline regions.
                - border_mask (numpy.ndarray): The mask for border regions.
        """
        # Preprocess
        image_cleaned, hairline_mask, border_mask = self._preprocess(image)

        # Inference
        class_summary = self._inference(image_cleaned)

        # Postprocess
        class_summary = self._postprocess(image, image_cleaned, class_summary)

        return class_summary, image_cleaned, hairline_mask, border_mask

    def evaluate(self, dataset_directory, enable_postprocess=True):
        """
        Evaluates the model on a dataset located in the specified directory, 
        calculating accuracy and generating a confusion matrix.

        Args:
            dataset_directory (str): Path to the dataset directory. The directory should 
            contain subdirectories for each class (e.g., "Benign", "Malignant").
            batch_size (int, optional): Number of images to process in a batch. Defaults to 32.

        Returns:
            tuple: A tuple containing:
                - accuracy (float): The accuracy of the model on the dataset.
                - conf_matrix (numpy.ndarray): The confusion matrix for the model's predictions.
        """
        # Load the dataset
        dataset = datasets.ImageFolder(root=dataset_directory)

        true_labels = []
        pred_labels = []

        def pil_to_opencv(image):
            
            if isinstance(image, Image.Image):
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image

        postprocess_flag = "With" if enable_postprocess else "Without"
        for image, label in tqdm(dataset, desc=f"Evaluate {postprocess_flag} Postprocessing"):
            
            image = pil_to_opencv(image)

            image_cleaned, _, _ = self._preprocess(image)

            class_summary = self._inference(image_cleaned)

            if enable_postprocess:
                class_summary = self._postprocess(image, image_cleaned, class_summary)

            predicted_class_id = class_summary["class"]

            true_labels.append(label)
            pred_labels.append(predicted_class_id)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)

        precision = precision_score(true_labels, pred_labels)
        
        return round(accuracy,3), round(precision,3)