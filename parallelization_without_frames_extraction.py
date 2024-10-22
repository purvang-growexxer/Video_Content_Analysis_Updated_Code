import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, FocalNetForImageClassification
import time
import concurrent.futures

def detect_nsfw_single_image(image_path, feature_extractor, model, device, transform, label_to_category):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Move tensor to device
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)  # Move inputs to device

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)

    label = model.config.id2label[predicted.item()]
    return image_path, label, confidence.item() * 100

def detect_nsfw_content(image_folder, output_file, max_workers=8):
    model_path = "MichalMlodawski/nsfw-image-detection-large"
    jpg_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.lower().endswith(".jpg")]

    if not jpg_files:
        print("🚫 No jpg files found in folder:", image_folder)
        return

    # Check for GPU availability and set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("💻 Using GPU (T4) for inference")
    else:
        device = torch.device('cpu')
        print("💻 GPU not available, using CPU")

    # Load model and move to the correct device
    feature_extractor = AutoProcessor.from_pretrained(model_path)
    model = FocalNetForImageClassification.from_pretrained(model_path).to(device)  # Move model to device
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    label_to_category = {
        "LABEL_0": "SAFE",
        "LABEL_1": "Questionable",
        "LABEL_2": "Unsafe"
    }

    # Parallel NSFW detection
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(detect_nsfw_single_image, image, feature_extractor, model, device, transform, label_to_category): image for image in jpg_files}
        for future in concurrent.futures.as_completed(future_to_image):
            try:
                image_file, label, confidence = future.result()
                results.append((image_file, label, confidence))
            except Exception as exc:
                print(f"Error processing {future_to_image[future]}: {exc}")

    # Save results to file in the desired format
    with open(output_file, 'w') as f:
        for jpg_file, label, confidence in results:
            category = label_to_category.get(label, "Unknown")
            f.write(f"File name: {jpg_file}\n")
            f.write(f"Model Label: {label}\n")
            f.write(f"NSFW Category: {category}\n")
            f.write(f"Confidence: {confidence:.2f}%\n")
            f.write("\n")  # Adds an extra line space between responses

    print(f"NSFW detection results saved to {output_file}")

# Main function to run the NSFW detection
def process_nsfw_detection(image_folder, nsfw_output_file):
    start_time = time.time()

    # Step 1: Detect NSFW content in the pre-extracted frames
    print("Starting NSFW detection...")
    nsfw_detection_start = time.time()
    detect_nsfw_content(image_folder, nsfw_output_file, max_workers=12)  # Adjust the number of workers based on testing
    nsfw_detection_end = time.time()
    print(f"NSFW detection completed in {nsfw_detection_end - nsfw_detection_start:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Total time taken for the NSFW detection process: {total_time:.2f} seconds.")

# Example usage
image_folder = '/content/bhool_bhulaiyaa_parallelization'  # Folder with pre-extracted frames
nsfw_output_file = '/content/bhool_bhulaiyaa_parallelization_without_frames_extraction.txt'

process_nsfw_detection(image_folder, nsfw_output_file)
