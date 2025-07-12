import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from models.resnet import resnet18, resnet50, resnet101


def load_and_preprocess_image(image_path, size=224):
    """
    Load and preprocess an image for ResNet inference
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_class(model, image_tensor, class_names=None):
    """
    Make prediction using the ResNet model
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities


def main():
    # Example class names (CIFAR-10 classes)
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create different ResNet models
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'ResNet-50': resnet50(num_classes=10),
        'ResNet-101': resnet101(num_classes=10)
    }
    
    # Create a dummy image tensor for demonstration
    # In practice, you would load a real image
    dummy_image = torch.randn(1, 3, 224, 224)
    
    print("ResNet Model Examples")
    print("=" * 50)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Make prediction
        predicted_class, confidence, probabilities = predict_class(model, dummy_image, cifar10_classes)
        
        print(f"Predicted class: {cifar10_classes[predicted_class]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        print("Top 3 predictions:")
        for i in range(3):
            print(f"  {cifar10_classes[top3_indices[i]]}: {top3_probs[i]:.4f}")
    
    # Test with different input sizes
    print("\n" + "=" * 50)
    print("Testing with different input sizes:")
    
    model = resnet18(num_classes=10)
    for size in [224, 256, 299]:
        dummy_input = torch.randn(1, 3, size, size)
        try:
            output = model(dummy_input)
            print(f"Input size {size}x{size}: Output shape {output.shape}")
        except Exception as e:
            print(f"Input size {size}x{size}: Error - {e}")


if __name__ == "__main__":
    main() 