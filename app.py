import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Define SimpleCNN class (copied from your training script)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 25, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 80 -> 40
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 40 -> 20
        
        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 20 -> 10
        
        # Conv block 4
        x = self.relu(self.conv4(x))
        x = self.pool(x)  # 10 -> 5
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Page title
st.title("🍎 Fruit Image Classifier")
st.write("Upload an image to classify it!")

# The 5 classes
classes = ["apple", "banana", "grape", "mango", "strawberry"]

# Load the model
@st.cache_resource
def load_model():
    model = torch.load('fruit_model.pt', map_location='cpu')
    model.eval()
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess the image (must match training: 80x80, not 224x224)
    transform = transforms.Compose([
        transforms.Resize((80, 80)),  # Changed to 80x80 to match your training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Show results
    st.success(f"**Prediction:** {classes[predicted.item()].upper()}")
    st.info(f"**Confidence:** {confidence.item() * 100:.2f}%")
    
    # Show all probabilities
    st.write("---")
    st.write("**All Class Probabilities:**")
    for i, class_name in enumerate(classes):
        prob = probabilities[0][i].item() * 100
        st.write(f"**{class_name.capitalize()}:** {prob:.2f}%")