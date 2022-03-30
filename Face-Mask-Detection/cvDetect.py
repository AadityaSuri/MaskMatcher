import cv2
from matplotlib import pyplot as plt
from main import CNN
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
# from cvCrop import cropFace


def isWearingMask(image_path):
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    classes = ("with_mask", "without_mask")

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    eye_data = cv2.CascadeClassifier("haarcascade_eye.xml")
    found = eye_data.detectMultiScale(img_gray, 1.1, 4)

    amount_found = len(found)

    if amount_found != 0:
        model = CNN()
        model.load_state_dict(torch.load("model.pth"))

        model.eval()

        img = Image.open(image_path).convert('RGB')
        x = transform(img)
        x = x.unsqueeze(0)

        output = model(x)
        pred = F.softmax(output, dim=1)
        return pred
    else:
        return -1

# print(isWearingMask('bg.png'))
