import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Hello This Project Made By: Omar Elzarka - Ahmed Elmehalawy - Mohamed Ghazal - Ahmed Fouad', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Press The Letter A To Start :)    ', (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
        
        cv2.imshow('Intelligent Systems Department - Image Processing Project', frame)
        if cv2.waitKey(25) == ord('a'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25) # Wait 25 Millsecond to Update Again
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
