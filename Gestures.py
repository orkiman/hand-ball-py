import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/root/PycharmProjects/hand_working4/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
cap = cv2.VideoCapture(0)
with GestureRecognizer.create_from_options(options) as recognizer:
    # The detector is initialized. Use it here.
    # ...
    # Use OpenCV’s VideoCapture to start capturing from the webcam.

    # Create a loop to read the latest frame from the camera using VideoCapture#read()

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # recognition_result = recognizer.recognize(mp_image)
        timestamps = cap.get(cv2.CAP_PROP_POS_MSEC)

        recognizer.recognize_async(mp_image, timestamps)
        # print_result(recognition_result)
        cv2.imshow('frame', cv2.flip(image, 1))
        # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()