from monai.transforms import Compose, LoadImage, ScaleIntensity, Resize, ToTensor

class CustomTransform:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transforms = Compose([
            LoadImage(image_only=True),
            ScaleIntensity(),
            Resize(self.target_size),
            ToTensor()
        ])

    def __call__(self, image):
        return self.transforms(image)