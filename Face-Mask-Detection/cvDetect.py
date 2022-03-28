import cv2
from matplotlib import pyplot as plt
from main import CNN
import torch
from torchvision import transforms
from PIL import Image
from cvCrop import cropFace

# returns 0 if mask is detected, 1 if not detected, -1 if person is not detected

def isWearingMask(image_path):
    print("started")
    transform_3 = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # image_path = "https___cdn.cnn.com_cnnnext_dam_assets_201107093022-02-centered-main-bar-final-biden.jpg"  # add original image path here

    isFace = cropFace(image_path)

    classes = ("with_mask", "without_mask")


    # img = cv2.imread(image_path_new)
    #
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # face_data = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    #
    # found = face_data.detectMultiScale(img_gray,
    #                                    minSize=(20, 20))
    # amount_found = len(found)

    if isFace != 0:
        image_path_new = "face.jpg"

        model = CNN()
        model.load_state_dict(torch.load("model.pth"))

        model.eval()

        img = Image.open(image_path_new)
        x = transform_3(img)
        x = x.unsqueeze(0)

        output = model(x)  # Forward pass
        pred = torch.argmax(output, 1)
        # print('Image predicted as', classes[pred])
        #
        # plt.subplot(1, 1, 1)
        # plt.imshow(img)
        # plt.show()

        return int(pred)
    else:
        # print("no face detected")
        # plt.subplot(1, 1, 1)
        # plt.imshow(Image.open(image_path))
        # plt.show()

        return -1

# print(isWearingMask("image-path-goes-here"))

