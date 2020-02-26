import pickle
import os 
import cv2
import face_recognition as fr 

dataset_path = os.path.join(os.getcwd(), 'Dataset')
encoding_file = os.path.join(dataset_path, 'data.pickle')

def train_face(name) :
    data = {}

    if os.path.isfile(encoding_file) :
        with open(encoding_file, 'rb') as f:
            data = pickle.load(f)

    encodings_list = []
    path = os.path.join(dataset_path, name)
    imgs = os.listdir(path)

    l = len(imgs)
    c = 1

    for i in imgs :
        img = fr.load_image_file(os.path.join(path, i))
        encoding = fr.face_encodings(img)[0]
        encodings_list.append(encoding)
        print(f'[INFO] Processed {c}/{l} images...')
        c += 1

    if name in data :
        data[name] += encodings_list
    else :
        data[name] = encodings_list

    with open(encoding_file, 'wb') as f :
        pickle.dump(data, f)


def capture_face() :
    name = input('Enter name : ')

    img_path = os.path.join(dataset_path, name)
    if not os.path.exists(img_path) :
        os.mkdir(img_path)

    v = cv2.VideoCapture(0)
    count = 1
    frame = 0

    cv2.namedWindow('Capturing Face', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Capturing Face', 1920, 1080)

    while True :
        ret, img = v.read()

        cv2.imshow('Capturing Face', img)
        
        if frame % 3 == 0 :
            save_location = os.path.join(img_path, f'{count}.jpg')
            cv2.imwrite(save_location, img)
            count += 1

        if cv2.waitKey(1) == 27 or count == 51:
            break 

        frame += 1

    v.release()
    cv2.destroyAllWindows()

    train_face(name)


def attendance_daalo() :
    if not os.path.isfile(encoding_file) :
        print('Kya re?! Joking aa? How to recognize without data??!')
        return False

    img = fr.load_image_file(os.path.join(os.getcwd(), 'class_img.jpg'))
    faces = fr.face_locations(img)
    img_encodings = fr.face_encodings(img, faces)  


capture_face()