import torch
import torch.nn as nn

# HuggingFace for ViT
from transformers import AutoFeatureExtractor, AutoModel

# Torchvision for CNN
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from utils.config import size_of_image

#transform for image dataloader
transform = T.Compose([
    T.Resize((size_of_image, size_of_image)),
    T.ToTensor()
])

class VisionEncoder(nn.Module):
    def __init__(self, 
                 backbone="vit", 
                 model_name="google/vit-base-patch16-224-in21k",
                 d_model=512,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.backbone_type = backbone

        if backbone == "vit":
            # HuggingFace ViT
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name).to(device)
            vision_dim = self.backbone.config.hidden_size

        elif backbone == "cnn":
            # Torchvision CNN (e.g., ResNet50)
            cnn = models.resnet50(pretrained=True)
            modules = list(cnn.children())[:-1]  # drop classification head
            self.backbone = nn.Sequential(*modules).to(device)
            vision_dim = cnn.fc.in_features

            # Standard transforms for CNN
            self.feature_extractor = T.Compose([
                T.Resize((size_of_image, size_of_image)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        else:
            raise ValueError("Unsupported backbone. Use 'vit' or 'cnn'.")

        # Projection to transformer hidden size
        self.projector = nn.Linear(vision_dim, d_model).to(device)


    def forward(self, images):
        """
        images: list of PIL Images or a batch tensor
        returns: projected embeddings [B, N, d_model]
        """
        if self.backbone_type == "vit":
            # HuggingFace ViT pipeline
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            outputs = self.backbone(**inputs)
            vision_embeds = outputs.last_hidden_state  # [B, N, vision_dim]

        elif self.backbone_type == "cnn":
            if isinstance(images, list):
                images = torch.stack([self.feature_extractor(img) for img in images])
            inputs = images.to(self.device)  # [B, 3, 224, 224]
            outputs = self.backbone(inputs).squeeze(-1).squeeze(-1)  # [B, vision_dim]
            vision_embeds = outputs.unsqueeze(1)  # [B, 1, vision_dim]

        return self.projector(vision_embeds)  # [B, N, d_model]
