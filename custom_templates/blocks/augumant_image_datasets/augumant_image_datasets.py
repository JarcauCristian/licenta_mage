# Variables {"image_column":{"type":"str","description":"The column were the image links are stored.","regex":"^.*$"},""transform": {"type": "drop_down", "description": "The transformation functions to apply to the images.", "values": ["Basic Augmentation", "Facial Recognition Augmentation", "Object Detection Augmentation", "Medical Imaging Augmentation", "Robust Augmentation"]}}

import io
import base64
import requests
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def get_transformation_pipeline(transformation: str) -> list:
    compose_list = []

    for t in transformation.split("->"):
        safe_builtins = {'None': None, 'True': True, 'False': False, 'dict': dict}
        name, params = t.split("(")[0], t.split("(")[1][:-1]
        try:
            params = eval(f'dict({params})', {'__builtins__': safe_builtins}, {})
        except SyntaxError as e:
            raise ValueError("Failed to parse parameters. Please check the format.") from e
        
        module = getattr(transforms, name)(**params)

        compose_list.append(module)

    return transforms.Compose(compose_list)


def convert_image_to_base64(image_path):
    response = requests.get(image_path)

    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_base64


def get_augmentation_object(augmentation_type: str):
    augmentation_type_map = {
        "Basic Augmentation": "RandomHorizontalFlip(p=0.5)->RandomVerticalFlip(p=0.5)->ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)->RandomRotation(degrees=15)",
        "Facial Recognition Augmentation": "RandomHorizontalFlip(p=0.5)->ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)->RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))",
        "Object Detection Augmentation": "RandomHorizontalFlip(p=0.5)->RandomCrop(size=256, pad_if_needed=True)->ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)",
        "Medical Imaging Augmentation": "RandomRotation(degrees=10)->RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5)->GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))",
        "Robust Augmentation": "RandomHorizontalFlip()->RandomRotation(degrees=30)->RandomPerspective(distortion_scale=0.6, p=0.5)->ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)->RandomErasing(p=0.1)"
    }

    if augmentation_type not in augmentation_type_map.keys():
        raise TypeError(f"Augmentation type can only be {list(augmentation_type_map.keys())}")

    return get_transformation_pipeline(augmentation_type_map[augmentation_type])


@transformer
def transform(data, *args, **kwargs):
    """
    Augments images in a DataFrame that are stored as Base64 encoded strings and adds them as new rows.
    
    Args:
        data: DataFrame with a column 'image_base64' that contains Base64 encoded images.
        args: Additional arguments (not used here).
        kwargs: Additional keyword arguments (not used here).

    Returns:
        DataFrame with original and augmented images in Base64 encoding.
    """

    image_column = kwargs.get("image_column")

    transform = kwargs.get("transform")

    if None in [image_column, transform]:
        raise ValueError("All kwargs are required (image_column, transform).")

    data[image_column] = data[image_column].apply(convert_image_to_base64)

    augmentation = get_augmentation_object(transform)

    def augment_image(base64_string):
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        img_transformed = augmentation(img)

        buffered = io.BytesIO()
        img_transformed.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_base64

    augmented_data = data.copy()
    
    augmented_data[image_column] = data[image_column].apply(augment_image)
    
    output_data = pd.concat([data, augmented_data], ignore_index=True)
    
    return output_data


@test
def test_output(output, *args) -> None:
    """
    Tests whether the output DataFrame is not None and verifies that it is twice the size of the input,
    assuming each image is augmented exactly once.
    """
    assert output is not None, 'The output is undefined'
    assert len(output) % 2 == 0, 'Output DataFrame should have an even number of rows (original + augmented)'

