{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.14","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"nvidiaTeslaT4","dataSources":[{"sourceId":1462296,"sourceType":"datasetVersion","datasetId":857191}],"dockerImageVersionId":30787,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":true}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-10-30T10:17:20.070447Z\",\"iopub.execute_input\":\"2024-10-30T10:17:20.071194Z\",\"iopub.status.idle\":\"2024-10-30T10:17:32.693420Z\",\"shell.execute_reply.started\":\"2024-10-30T10:17:20.071148Z\",\"shell.execute_reply\":\"2024-10-30T10:17:32.692469Z\"}}\n!pip install pycocotools\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-10-30T10:17:32.695821Z\",\"iopub.execute_input\":\"2024-10-30T10:17:32.696568Z\",\"iopub.status.idle\":\"2024-10-30T10:17:51.943461Z\",\"shell.execute_reply.started\":\"2024-10-30T10:17:32.696520Z\",\"shell.execute_reply\":\"2024-10-30T10:17:51.942525Z\"}}\nimport torch\nfrom torchvision import datasets, transforms, models\nfrom torch.utils.data import DataLoader\nimport torch.nn as nn\nimport torch.nn.functional\nimport torch.optim as optim\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom torchvision.transforms import functional as F\n\n\n# Define the dataset transform\ntransform = transforms.Compose([\n    transforms.ToTensor(),\n])\n\n# Custom collate function to pad images in the batch to the same size\ndef collate_fn(batch):\n    # Separate images and targets\n    images, targets = zip(*batch)\n    \n    # Find max height and width in the batch\n    max_height = max(img.shape[1] for img in images)\n    max_width = max(img.shape[2] for img in images)\n    \n    # Pad each image to the max height and width\n    padded_images = []\n    for img in images:\n        # Pad to (max_height, max_width)\n        padded_img = torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1]))\n        padded_images.append(padded_img)\n    \n    # Stack images and return targets as they are\n    return torch.stack(padded_images, dim=0), targets\n\n# Now, load your dataset and dataloader\n# Load the dataset\ntrain_dataset = datasets.CocoDetection(root='/kaggle/input/coco-2017-dataset/coco2017/train2017', \n                                       annFile='/kaggle/input/coco-2017-dataset/coco2017/annotations/person_keypoints_train2017.json',\n                                       transform=transform)\ntrain_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-10-30T10:17:51.944968Z\",\"iopub.execute_input\":\"2024-10-30T10:17:51.945509Z\",\"iopub.status.idle\":\"2024-10-30T10:17:51.968875Z\",\"shell.execute_reply.started\":\"2024-10-30T10:17:51.945463Z\",\"shell.execute_reply\":\"2024-10-30T10:17:51.967579Z\"}}\nfrom torchvision.models import resnet50, ResNet50_Weights\n\nclass ObjectCountingModel(nn.Module):\n    def __init__(self, num_classes):\n        super(ObjectCountingModel, self).__init__()\n        \n        # Backbone: ResNet-50 with new weights syntax\n        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)\n        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer\n        \n        # Classification Branch\n        self.classification_branch = nn.Sequential(\n            nn.Linear(2048, 512),\n            nn.ReLU(),\n            nn.Linear(512, num_classes),\n            nn.Sigmoid()\n        )\n        \n        # Density Branch\n        self.density_branch = nn.Sequential(\n            nn.Conv2d(2048, 512, kernel_size=3, padding=1),\n            nn.ReLU(),\n            nn.Conv2d(512, num_classes, kernel_size=1)  # Separate density map for each category\n        )\n        \n        # Weight Modulation Layer (RLC)\n        self.modulation_layer = nn.Sequential(\n            nn.Linear(2048, 512),\n            nn.ReLU(),\n            nn.Linear(512, num_classes)\n        )\n\n    def forward(self, x):\n        features = self.backbone(x)\n        \n        class_preds = self.classification_branch(features)\n        \n        density_maps = self.density_branch(features.unsqueeze(-1).unsqueeze(-1))\n        \n        modulated_weights = self.modulation_layer(features)\n        \n        return class_preds, density_maps, modulated_weights\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-10-30T10:17:51.970227Z\",\"iopub.execute_input\":\"2024-10-30T10:17:51.970510Z\",\"iopub.status.idle\":\"2024-10-30T10:18:03.339677Z\",\"shell.execute_reply.started\":\"2024-10-30T10:17:51.970480Z\",\"shell.execute_reply\":\"2024-10-30T10:18:03.338498Z\"}}\n# Replace `cu117` with the CUDA version compatible with your environment\n!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-10-30T10:18:03.342307Z\",\"iopub.execute_input\":\"2024-10-30T10:18:03.342645Z\",\"iopub.status.idle\":\"2024-10-30T10:18:03.627524Z\",\"shell.execute_reply.started\":\"2024-10-30T10:18:03.342610Z\",\"shell.execute_reply\":\"2024-10-30T10:18:03.626531Z\"}}\nfor images, targets in train_loader:\n    print(targets[:10])  # This will show the structure of `targets` and help us identify the issue\n    break \n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-10-30T10:18:03.629187Z\",\"iopub.execute_input\":\"2024-10-30T10:18:03.629799Z\",\"iopub.status.idle\":\"2024-10-30T10:18:06.544114Z\",\"shell.execute_reply.started\":\"2024-10-30T10:18:03.629749Z\",\"shell.execute_reply\":\"2024-10-30T10:18:06.542748Z\"}}\nimport torch.optim as optim\nimport torch\nimport torch.nn as nn\n\nprint(torch.cuda.is_available())  # Check if CUDA is available\n\n# Define loss functions\nclassification_loss = nn.CrossEntropyLoss()  # For multi-class classification\ndensity_loss = nn.MSELoss()  # For density map loss\n\n# Initialize model, optimizer, and scheduler\nmodel = ObjectCountingModel(num_classes=80)\noptimizer = optim.Adam(model.parameters(), lr=1e-4)\n\n# Function to create density map from bounding boxes\ndef create_density_map(bboxes, image_size):\n    density_map = torch.zeros(image_size)  # Initialize a density map\n    for bbox in bboxes:\n        x, y, w, h = bbox  # Unpack bounding box\n        # Ensure the indices are integers\n        x, y, w, h = int(x), int(y), int(w), int(h)\n        # Create a Gaussian distribution centered on the bounding box\n        # For simplicity, you can just increment the area or use more complex logic\n        density_map[y:y+h, x:x+w] += 1  # Mark the area in the density map\n    return density_map\n\n# Training Loop\ndef train(model, train_loader, optimizer, epochs):\n    model.train()\n    for epoch in range(epochs):\n        total_loss = 0\n        for images, targets in train_loader:\n            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')\n\n            # Unpack targets\n            category_ids_list = []\n            density_targets_list = []\n\n            for target_list in targets:\n                for t in target_list:\n                    category_ids_list.append(t['category_id'])\n                    # Extract bounding boxes and create density map\n                    if 'bbox' in t:\n                        density_targets_list.append(t['bbox'])  # Keep the bbox for density map creation\n\n            # Convert category ids to tensor\n            category_ids_tensor = torch.tensor(category_ids_list).to('cuda' if torch.cuda.is_available() else 'cpu').long()\n\n            # Create a density map for the images\n            density_targets_tensor = create_density_map(density_targets_list, images.size()[2:])  # Assuming images are in [N, C, H, W]\n            density_targets_tensor = density_targets_tensor.to('cuda' if torch.cuda.is_available() else 'cpu').float()\n\n            # Ensure the model is on the GPU\n            model = model.to('cuda')\n            # Forward pass\n            class_preds, mod_weights= model(images)\n\n            # Calculate classification loss\n            class_loss = classification_loss(class_preds, category_ids_tensor)\n\n            # Calculate density loss\n            density_loss_val = density_loss(density_maps, density_targets_tensor)\n\n            # Total loss\n            loss = class_loss + density_loss_val  # Combine losses\n\n            # Backpropagation and optimization\n            optimizer.zero_grad()\n            loss.backward()\n            optimizer.step()\n\n            total_loss += loss.item()\n\n        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}')\n\ntrain(model, train_loader, optimizer, epochs=10)\n\n# %% [code]\n","metadata":{"_uuid":"e165c748-03c2-4563-b9d9-a25f80e3aca0","_cell_guid":"19d7a7be-0f00-4225-801c-0419d0079322","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}