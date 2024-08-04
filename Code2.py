import cv2 as cv
import numpy as np

#ورودی ویدیو را از دوربین دریافت می‌کنیم
capture = cv.VideoCapture(0)

#مدل های تشخیص چهره و چشم را بارگذاری می‌کنیم.

face_cascade = cv.CascadeClassifier('D:/University/Karshenashi/Image Processing/Exercise/Project/RealSlope/models/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('D:/University/Karshenashi/Image Processing/Exercise/Project/RealSlope/models/haarcascade_eye.xml')

#در حلقه‌ای بی‌نهایت، فریم فعلی از ویدیو را خو
# انده و سپس به سطح خاکستری تبدیل می‌کنیم.

while True:
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #سپس، چهره‌ها را در تصویر سطح خاکستری تشخیص می‌دهیم.
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    x, y, w, h = 0, 0, 0, 0

    #برای هر چهره، یک مستطیل و یک دایره رسم می‌کنیم.
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(frame, (x + int(w * 0.5), y +
                          int(h * 0.5)), 4, (0, 255, 0), -1)
    #سپس، چشم‌ها را در ناحیه صورت تشخیص می‌دهیم.
    eyes = eye_cascade.detectMultiScale(gray[y:(y + h), x:(x + w)], 1.1, 4)
    #برای هر چشم، یک مستطیل رسم می‌کنیم و مختصات آن را ذخیره می‌کنیم.
    index = 0
    eye_1 = [None, None, None, None]
    eye_2 = [None, None, None, None]
    for (ex, ey, ew, eh) in eyes:
        if index == 0:
            eye_1 = [ex, ey, ew, eh]
        elif index == 1:
            eye_2 = [ex, ey, ew, eh]
        cv.rectangle(frame[y:(y + h), x:(x + w)], (ex, ey),
                     (ex + ew, ey + eh), (0, 0, 255), 2)
        index = index + 1
    #در صورت وجود دو چشم، زاویه انحراف چشم‌ها را محاسبه و نمایش می‌دهیم.
    if (eye_1[0] is not None) and (eye_2[0] is not None):
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
            int(left_eye[1] + (left_eye[3] / 2)))

        right_eye_center = (
            int(right_eye[0] + (right_eye[2] / 2)),
            int(right_eye[1] + (right_eye[3] / 2)))

        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        #محاسبه مختصات مرکز هر چشم و محاسبه تفاوت مختصات X و Y بین آن‌ها.
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        #محاسبه زاویه انحراف چشم‌ها با استفاده از تابع تانژانت و تبدیل آن به درجات.
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y

        angle = np.arctan(delta_y / delta_x)

        angle = (angle * 180) / np.pi

        #در صورتی که زاویه بیشتر از ۱۰ درجه باشد، پیامی شامل مقدار زاویه را به تصویر اضافه می‌کنیم.
        if angle > 10:
            cv.putText(frame, 'RIGHT TILT :' + str(int(angle)) + ' degrees',
                       (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 2, cv.LINE_4)
        elif angle < -10:
            cv.putText(frame, 'LEFT TILT :' + str(int(angle)) + ' degrees',
                       (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 2, cv.LINE_4)
        else:
            cv.putText(frame, 'STRAIGHT :', (20, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 2, cv.LINE_4)

    #در نهایت، تصویر را نمایش می‌دهیم.
    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break
#پس از اتمام، منابع را آزاد می‌کنیم.
capture.release()
cv.destroyAllWindows()