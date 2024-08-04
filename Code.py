import dlib
import cv2
import math


def calculate_slope(landmarks):
    left_eye = landmarks.part(36)
    right_eye = landmarks.part(45)
    slope = math.degrees(math.atan((right_eye.y - left_eye.y) / (right_eye.x - left_eye.x)))
    return slope


def draw_slope_text(image, slope):
    text = "Slope: {:.2f}".format(slope)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Unable to open the webcam.")
        return

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:/University/Karshenashi/Image Processing/Exercise/Project/RealSlope/models/shape_predictor_68_face_landmarks.dat")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            slope = calculate_slope(landmarks)
            draw_slope_text(frame, slope)

            # رسم مستطیل دور چهره
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # رسم نقاط مختلف صورت (نقاط چشم)
            for i in range(36, 48):
                x, y = landmarks.part(i).x, landmarks.part(i).y

                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow('RealSlope', frame)
        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

