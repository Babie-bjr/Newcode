import cv2
import os

name = 'Mangosteen'  # Change name to Mangosteen
# Create directory if it doesn't exist
directory = f'dataset/{name}/'
if not os.path.exists(directory):
    os.makedirs(directory)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Press space to take a photo", 500, 300)

img_counter = 1
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Press space to take a photo", frame)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = f"{directory}image_{img_counter}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
