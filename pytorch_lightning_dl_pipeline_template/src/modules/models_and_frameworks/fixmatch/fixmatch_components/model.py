from torchvision.models import efficientnet_b0, resnet18, resnet34, resnet50, convnext_tiny
import torch
import torch.nn as nn


class FixMatchModel:
    def __new__(cls, config):
        if config.name == "EfficientNet":
            return EfficientNetModel(config)
        elif config.name == "ResNet-18":
            return ResNet18Model(config)
        elif config.name == "ResNet-34":
            return ResNet34Model(config)
        elif config.name == "ResNet-50":
            return ResNet50Model(config)
        elif config.name == "ConvNext":
            return  ConvNextModel(config)
        else:
            raise ValueError(
                f"Invalid model name: {config.name}. "
                f"Supported model names are 'EfficientNet', "
                f"'ResNet-18', 'ResNet-34', and 'ResNet-50'."
            )

class ConvNextModel(torch.nn.Module):
    def __init__(self, config):
        super(ConvNextModel, self).__init__()
        self.model = convnext_tiny()
        self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        self.model.classifier[2] = nn.Linear(768, 2)
       
    def forward(self, model_input):
        model_output = self.model(model_input)
        return model_output

class EfficientNetModel(torch.nn.Module):
    def __init__(self, config):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0()
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        if model_input.shape[1] == 1:  # If input is grayscale
            model_output = self.model(model_input.repeat(1, 3, 1, 1))
        else:
            model_output = self.model(model_input)
        return model_output

class ResNet18Model(torch.nn.Module):
    def __init__(self, config):
        super(ResNet18Model, self).__init__()
        self.model = resnet18()
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output

class ResNet34Model(torch.nn.Module):
    def __init__(self, config):
        super(ResNet34Model, self).__init__()
        self.model = resnet34()
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output

class ResNet50Model(torch.nn.Module):
    def __init__(self, config):
        super(ResNet50Model, self).__init__()
        self.model = resnet50()
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output