import cv2

def cropFace(image_path):
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade1 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    faces = face_cascade1.detectMultiScale(gray, 1.1, 4)

    pd = 75
    for (x, y, w, h) in faces:

        faces = img[max(0, y - pd):min(height, y + h + pd), max(0, x - pd):min(width, x + w + pd)]
        cv2.imwrite('face.jpg', faces)

    if len(faces) != 0:
        return True
    else:
        return False



