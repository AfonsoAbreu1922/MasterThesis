import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from data_loader import LungNoduleDataset
from classifier import LungNoduleClassifier

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert to tensor
])

# Example dataset (Replace with real dataset paths)
image_paths = ["./data/image_1.png", "./data/image_2.png"]
labels = [0, 1]  # Example labels: 0 (benign), 1 (malignant)

# Train-test split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# Create PyTorch datasets
train_dataset = LungNoduleDataset(train_paths, train_labels, transform=transform)
val_dataset = LungNoduleDataset(val_paths, val_labels, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Initialize model
model = LungNoduleClassifier(lr=1e-4)

# Trainer configuration
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    log_every_n_steps=10
)

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the model
trainer.save_checkpoint("best_model.ckpt")

# Define a function for prediction
def predict(model_path, image_path):
    model = LungNoduleClassifier.load_from_checkpoint(model_path)
    model.eval()

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    logits = model(image)
    prob = torch.sigmoid(logits).item()
    return "Malignant" if prob > 0.5 else "Benign"

# Example inference
prediction = predict("best_model.ckpt", "./data/image_1.png")
print(f"Prediction: {prediction}")
