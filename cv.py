import torch
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Function to load the image and preprocess it for the model
def load_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Convert BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define transformations for the image
    transform = T.Compose([
        T.ToTensor(),  # Convert the image to tensor
    ])

    # Apply transformations
    image_tensor = transform(image_rgb).unsqueeze(0)  # Add batch dimension
    return image_tensor, orig_image

# Function to load the PyTorch model
def load_model(model_path):
    # Load the pre-trained model from the .pth file
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to draw the bounding boxes and labels
def draw_boxes(orig_image, boxes, labels, scores, confidence_threshold=0.5):
    # Loop through the boxes, labels, and scores and draw them on the image
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:  # Filter out low-confidence detections
            x1, y1, x2, y2 = box.astype(int)  # Get box coordinates as integers

            # Draw bounding box on the image (color: green, thickness: 2)
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label text (label ID and confidence score)
            label_text = f'{label}: {score:.2f}'

            # Put the label text above the bounding box
            cv2.putText(orig_image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return orig_image

# Function to display the image using OpenCV or Matplotlib
def display_image(image, use_matplotlib=False):
    if use_matplotlib:
        # Convert BGR to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide axis
        plt.show()
    else:
        # Display the image using OpenCV
        cv2.imshow('Detected Objects', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main function to load model, run inference, and display results
def main(image_path, model_path, confidence_threshold=0.5, use_matplotlib=False):
    # Step 1: Load the model
    model = load_model(model_path)

    # Step 2: Load and preprocess the image
    image_tensor, orig_image = load_image(image_path)

    # Step 3: Perform inference (no gradient calculation needed)
    with torch.no_grad():
        predictions = model(image_tensor)

    # Step 4: Extract the bounding boxes, labels, and scores
    boxes = predictions[0]['boxes'].cpu().numpy()  # Bounding boxes
    labels = predictions[0]['labels'].cpu().numpy()  # Labels
    scores = predictions[0]['scores'].cpu().numpy()  # Confidence scores

    # Step 5: Draw the boxes and labels on the image
    result_image = draw_boxes(orig_image, boxes, labels, scores, confidence_threshold)

    # Step 6: Display the image
    display_image(result_image, use_matplotlib)

# Example usage:
if __name__ == "__main__":
    image_path = "your_image.jpg"  # Path to your image
    model_path = "model.pth"       # Path to your .pth model
    confidence_threshold = 0.5     # Confidence threshold for displaying boxes
    use_matplotlib = False         # Set to True if you prefer Matplotlib display

    # Run the main function
    main(image_path, model_path, confidence_threshold, use_matplotlib)
