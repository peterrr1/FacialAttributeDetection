import torch
from operator import itemgetter 
from torchvision.transforms import v2
from PIL import Image
import io

class_names = ['5 o Clock Shadow', 'Arched Eyebrows', 'Attractive', 'Bags Under Eyes', 'Bald', 'Bangs',
                'Big Lips', 'Big Nose', 'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows',
                  'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair', 'Heavy Makeup', 'High Cheekbones',
                    'Male', 'Mouth Slightly Open', 'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin',
                      'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns', 'Smiling', 'Straight Hair', 'Wavy Hair',
                        'Wearing Earrings', 'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace', 'Wearing Necktie', 'Young\n']

classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(1280, 128),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(128, 64),
    torch.nn.BatchNorm1d(64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(64, 40)
)

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')

model.classifier = classifier

model.load_state_dict(torch.load('models/MOBILE_NET_V2_1.pth'))
model.eval()


def transform_image(image_bytes):
    my_transforms = v2.Compose([
            v2.Resize(size=(234, 234)),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # Normalization for pretrained mobilenet: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    return outputs

def predict(file):
    image_bytes = file.read()
    sigmoid = torch.nn.Sigmoid()
    result = sigmoid(get_prediction(image_bytes=image_bytes)) > 0.5
    indices = torch.nonzero(result).tolist()
    predictions = list()
    for i in indices:
        predictions.append(i[1])
    predictions = itemgetter(*predictions)(class_names)
    return predictions

