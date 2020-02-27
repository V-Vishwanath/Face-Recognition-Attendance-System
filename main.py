import os 
import cv2 
import pickle
import sqlite3
from time import asctime, sleep
import face_recognition as fr

curr_dir = os.getcwd()
attendance_path = os.path.join(curr_dir, 'Attendance')
dataset_path = os.path.join(curr_dir, 'Dataset')
encoding_file = os.path.join(dataset_path, 'data.pickle')

if not os.path.exists(attendance_path) :
    os.mkdir(attendance_path)

if not os.path.exists(dataset_path) :
    os.mkdir(dataset_path)

get_month = {'01' : 'Jan',
'02' : 'Feb',
'03' : 'Mar',
'04' : 'Apr',
'05' : 'May',
'06' : 'Jun',
'07' : 'Jul',
'08' : 'Aug',
'09' : 'Sep',
'10' : 'Oct',
'11' : 'Nov',
'12' : 'Dec',}

def load_data() :
    if not os.path.isfile(encoding_file) :
        print('\tFile data.pickle is missing. Please restore it and try again!\n')
        return False

    global Attendace, known_encodings, known_faces, student_name

    with open(encoding_file, 'rb') as f :
        data = pickle.load(f)

    Attendace = {}
    student_name = {}
    known_encodings = []
    known_faces = []

    for i in data :
        known_encodings += data[i]
        name, _id = i.split('+')
        known_faces += ((_id + ' ')*len(data[i])).split()
        Attendace[_id] = 0
        student_name[_id] = name

    return True 


def train_face(reg_no, name) :
    data = {}

    if os.path.isfile(encoding_file) :
        with open(encoding_file, 'rb') as f:
            data = pickle.load(f)

    encodings_list = []
    path = os.path.join(dataset_path, reg_no)
    imgs = os.listdir(path)

    l = len(imgs)
    c = 1

    print('\n')
    print(f'  [INFO] Learning to recognize {name}...')
    for i in imgs :
        img = fr.load_image_file(os.path.join(path, i))
        try :
            encoding = fr.face_encodings(img)[0] 
            encodings_list.append(encoding)
            print(f'  [INFO] Processed {c}/{l} images...')
            c += 1
        except :
            pass 

    print('\n')

    if reg_no in data :
        data[name + '+' + reg_no] += encodings_list
    else :
        data[name + '+' + reg_no] = encodings_list

    with open(encoding_file, 'wb') as f :
        pickle.dump(data, f)

    load_data()


def capture_face() :
    print('\n')
    reg_no = input('\tEnter ID : ').upper()
    name = input('\tEnter name : ').title()

    img_path = os.path.join(dataset_path, reg_no)
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

    train_face(reg_no, name)


def capture_class() :
    v = cv2.VideoCapture(0)
    cv2.namedWindow('Class Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Class Image', 1920, 1080)

    ret, img = v.read()
    cv2.imshow('Class Image', img)
    cv2.imwrite('class_img.jpg', img)
    cv2.waitKey(100)

    v.release()
    cv2.destroyAllWindows()


def mark_faces() :
    if len(known_faces) is 0 :
        load_data()

    # capture_class()

    img = fr.load_image_file(os.path.join(curr_dir, 'class_img.jpg'))
    faces = fr.face_locations(img)
    encodings = fr.face_encodings(img, faces)  

    for encoding in encodings :
        result = fr.compare_faces(known_encodings, encoding, tolerance=0.4)

        _id = ''
        if True in result :
            index = result.index(True)
            _id = known_faces[index]

        if _id != '' :
            Attendace[_id] += 1


def create_table(db_name, table_name):
    database = sqlite3.connect(db_name)
    cursor = database.cursor()

    cursor.execute(f'SELECT count(name) FROM sqlite_master WHERE type="table" AND name="{table_name}"')

    if cursor.fetchone()[0] != 1 :
        cursor.execute(f'''CREATE TABLE {table_name} (
            ID TEXT PRIMARY KEY NOT NULL,
            Name TEXT NOT NULL,
            Attendance BIT 
            );''')

        for i in student_name :
            cursor.execute(f'INSERT INTO {table_name}(ID, Name) VALUES ("{i}", "{student_name[i]}")')


    database.commit()
    database.close()


def update_attendance(db_name, table_name) :
    database = sqlite3.connect(db_name)
    cursor = database.cursor()

    for i in Attendace :
        if Attendace[i] > 3 :
            cursor.execute(f'UPDATE {table_name} SET Attendance=1 WHERE ID="{i}"')

    database.commit()
    database.close()


def put_attendance() :
    load_data()

    formatted_time = asctime()

    date = formatted_time[8:10]
    month = formatted_time[4:7]
    year = formatted_time[-4:]

    if int(date) < 10 :
        date = '0' + str(int(date))
    
    date = 'Day' + date

    year_path = os.path.join(attendance_path, year)
    db_path = os.path.join(year_path, month+'.db')
    
    if not os.path.exists(year_path) :
        os.mkdir(year_path)

    if not os.path.isfile(db_path) :
        create_table(db_path, date)

    count = 0
    while count < 5 :
        mark_faces()
        count += 1
        sleep(1)

    update_attendance(db_path, date)


def check_database() :
    print('\n')
    date = input('\tEnter date (dd/mm/yyyy) : ')

    day = 'Day' + date[:2]
    month = get_month[date[3:5]]
    year = date[6:]

    year_path = os.path.join(attendance_path, year)
    db_path = os.path.join(year_path, month+'.db')

    if (not os.path.exists(year_path)) or (not os.path.isfile(db_path)) :
        print('Sorry! Data does not exist!\n')
        return False, None, None  

    return True, db_path, day  

def get_attendance() :
    available, path, day = check_database()
    
    if not available :
        return False 
    
    database = sqlite3.connect(path)
    cursor = database.cursor()

    data = cursor.execute(f'SELECT * FROM {day}')

    print('\n')
    print('-------------------------------------------------------------------')
    print('|    Roll Number    |       Student Name       |    Attendance    |')
    print('-------------------------------------------------------------------')
    print('|' + (' ' * 19) + '|' + (' ' * 26) + '|' + (' ' * 18) + '|')
    for i in data :
        print(f'|     {i[0]}     |', end='')

        l = len(i[1])
        spaces = int((26 - l) / 2)
        print((' ' * spaces) + i[1] + (' ' * (26 - spaces - l)) + '|', end='')

        if i[2] == None :
            print('      Absent      |')
        else :
            print('      Present     |')

        print('|' + (' ' * 19) + '|' + (' ' * 26) + '|' + (' ' * 18) + '|')
            
    print('-------------------------------------------------------------------\n')


    database.commit()
    database.close()


def check_attendance() :
    available, path, day = check_database()

    if not available :
        return False 

    _id = input('\tEnter ID : ').upper()
    print()

    if _id not in student_name :
        print('\tSorry! Given ID does not exist! Please register the student and try again!\n')
        return False 

    database = sqlite3.connect(path)
    cursor = database.cursor()

    data = cursor.execute(f'SELECT Attendance FROM {day} WHERE ID="{_id}"')
    for i in data :
        if i[0] == None :
            print(f'  [INFO] {student_name[_id]} was Absent')
        else :
            print(f'  [INFO] {student_name[_id]} was Present')

    print('\n')

    database.commit()
    database.close()


    
if __name__ == "__main__" :
    Attendace = {}
    student_name = {}
    known_faces = []
    known_encodings = []

    print('\n')
    while True :
        choice = input('''    Choose operation :

        1) Add new student
        2) Capture Attendance
        3) View Class Attendance
        4) Check Attendance With ID
        5) Exit
        
        Your choice : ''')

        if choice == '1' :
            capture_face()

        elif choice == '2' :
            put_attendance()

        elif choice == '3' :
            get_attendance()

        elif choice == '4' :
            check_attendance()

        elif choice == '5' :
            break


