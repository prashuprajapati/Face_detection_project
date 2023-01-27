mport cv2
import cvzone

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('Cascade_face_dection.xml')

filter_img = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)
print(filter_img.shape, type(filter_img))

while True:
    _, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_img)
    for (start, end, width, height) in faces:
        # cv2.rectangle(frame,(start, end), (start+width, end+height), (0, 255, 0), 2)
        filter_resize = cv2.resize(filter_img, (int(width * 1.5), int(height * 1.5)))
        frame = cvzone.overlayPNG(frame, filter_resize, [start - 45, end - 75])
    cv2.imshow('Snapchat_feature practice', frame)
    cv2.waitKey(1)
    # if cv2.waitKey(1) and ord('q'):
    #      continue