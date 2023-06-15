import torch
import torch.nn as nn
# include whatever other imports you need here

class SymbolClassifier(nn.Module):
   # Your network definition goes here
   def __init__(self):
      super(SymbolClassifier, self).__init__()

      self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
      self.bn1 = nn.BatchNorm2d(6)
      self.relu = nn.ReLU()
      self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

      self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1)
      self.bn2 = nn.BatchNorm2d(10)

      self.fc1 = nn.Linear(in_features=12*12*10,out_features=120)
      self.fc2 = nn.Linear(in_features=120,out_features=84)
      self.fc3 = nn.Linear(in_features=84,out_features=5)  

   def forward(self, x):
      # Input has dimensions (B is batch size):
      # B x  1 x 48 x 48
      x = self.conv1(x)
      # B x  6 x 48 x 48
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      # B x  6 x 24 x 24

      x = self.conv2(x)
      # B x 10 x 24 x 24
      x = self.bn2(x)
      x = self.relu(x)
      x = self.maxpool(x)
      # B x 10 x 12 x 12

      x = x.view(x.size(0), -1)
      # Flattened to B x 10*12*12
      x = self.fc1(x)
      x = self.relu(x)
      # B x 120
      x = self.fc2(x)
      x = self.relu(x)
      # B x 84
      x = self.fc3(x)
      # B x 5
      return x

def classify(images):
   # Determine which device the input tensor is on
   device = torch.device("cuda" if images.is_cuda else "cpu")

   model = SymbolClassifier()
   # Move to same device as input images
   model = model.to(device)
   # Load network weights
   model.load_state_dict(torch.load('weights.pkl',map_location=torch.device(device)))
   # Put model in evaluation mode
   model.eval()

   # Optional: do whatever preprocessing you do on the images
   # if not included as tranformations inside the model

   with torch.no_grad():
       # Pass images to model, get back logits or probabilities
       output = model(images)

   # Select class with highest probability for each input
   predicted_classes = torch.argmax(output, 1)

   # Return predicted classes
   return predicted_classes
