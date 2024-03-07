import cv2
import mediapipe as mp
import math
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands




# last_ball_x = 0
# last_ball_y = 0
last_ball = {'x': 0,
             'y': 0}
class game:
    last_hand=None
    # def __init__(self):
    #     last_hand=None
g=game
class Hand:
    id = -1
    x, y = 0, 0
    first_hand = True

class Ball:
    # x, y = 300, 0
    # radius = 30
    # speed = {'x': 0, 'y': 1}
    def __init__(self):
        self.init_ball()

    def init_ball(self):
        self.x, self.y = 300, 0
        self.radius = 30
        self.speed = {'x': 0, 'y': 1}


ball = Ball()
last_hand = Hand()


def draw_ball(image):
    image_height, image_width = image.shape[:2]
    cv2.circle(image, (ball.x,int (ball.y)), ball.radius, (0, 255, 0), -1)
    ball.x = np.clip(ball.x + ball.speed['x'], 0 + ball.radius, image_width - ball.radius)
    ball.y = np.clip(ball.y + ball.speed['y'], 0 + ball.radius, image_height - ball.radius)

    ball.speed['y'] += 0.1


def check_hit(current_hand):

    image_height, image_width = image.shape[:2]
    current_hand_x = (image_width * current_hand.x)
    current_hand_y = (image_height * current_hand.y)
    # print(ball.x, ball.y, current_hand_x, hand_y)
    if math.sqrt((current_hand_x - ball.x) ** 2 + (current_hand_y - ball.y) ** 2) < ball.radius:
        # print('******')
        if not last_hand.first_hand:
            x_speed = int((current_hand_x-last_hand.x))
            y_speed = int((current_hand_y-last_hand.y))
            ball.speed['x'] = np.clip(x_speed,-5,5)
            ball.speed['y'] = np.clip(y_speed,-5,5)
            # print(current_hand_x,last_hand.x,x_speed,y_speed)
            # print(ball.speed)

    last_hand.x = current_hand_x
    last_hand.y = current_hand_y
    last_hand.first_hand = False





class HandsManager:
    hands=[]
# def draw_ball(image, x, y, last_ball):
#     # draw the ball in the tip of the finger
#     image_height, image_width = image.shape[:2]
#     cv2.circle(image, (int(image_width * x) , int(image_height * y)), 15, (0, 255, 0), -1)
#     return last_ball



def checkHit(image, currentHand, last_hand):
    if(last_hand is None):
        return
    image_height, image_width = image.shape[:2]
    scale = 2000
    x1 = int(image_width * last_hand.x)
    y1 = int(image_height * last_hand.y)
    x2 = int(image_width * currentHand.x + (currentHand.x-last_hand.x) * scale)
    y2 = int(image_height * currentHand.y + (currentHand.y-last_hand.y) * scale)
    cv2.arrowedLine(image,(x1, y1), (x2, y2),(255, 0, 0), 2)
    # image, start_point, end_point,
    # color, thickness


tst = 1

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
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
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # if results.multi_handedness:
        #     for handed in results.multi_handedness:
        #         print(handed)
        draw_ball(image)
        if results.multi_hand_landmarks:
            counter = 0
            # print(tst)
            # tst += 1
            for hand_landmarks in results.multi_hand_landmarks :
                counter+=1
                # print(hand_landmarks.landmark[8], type(results.multi_handedness[0]))
                # print(hand_landmarks.landmark[8], results.multi_handedness[0].label)
                # print(results)
                # print(type(results))

                currentHand = hand_landmarks.landmark[8]
                x = currentHand.x # 8 = finger tip
                y = currentHand.y
                # checkHit(image, currentHand, g.last_hand)
                check_hit(currentHand)
                g.last_hand = hand_landmarks.landmark[8]
                # cv2.circle(image, ((int)(image_width * x), (int)(image_height * y)), 50, (0, 255, 0), -1)
                # last_ball=draw_ball(image, x, y, last_ball)

                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        # print(cv2.getWindowImageRect('Frame'))
        #     print(counter)
        #     exit(0)
        # cv2.resizeWindow("frame", 999, 999)

        cv2.imshow('frame', cv2.flip(image, 1))
        # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        elif key == 32:
            ball.init_ball()
            print(key)
cap.release()


# # For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#         static_image_mode=True,
#         max_num_hands=2,
#         min_detection_confidence=0.5) as hands:
#     for idx, file in enumerate(IMAGE_FILES):
#         # Read an image, flip it around y-axis for correct handedness output (see
#         # above).
#         image = cv2.flip(cv2.imread(file), 1)
#         # Convert the BGR image to RGB before processing.
#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#         # Print handedness and draw hand landmarks on the image.
#         print('Handedness:', results.multi_handedness)
#         if not results.multi_hand_landmarks:
#             continue
#         image_height, image_width, _ = image.shape
#         annotated_image = image.copy()
#         for hand_landmarks in results.multi_hand_landmarks:
#             print('hand_landmarks:', hand_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#             )
#             mp_drawing.draw_landmarks(
#                 annotated_image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#         cv2.imwrite(
#             '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#         # Draw hand world landmarks.
#         if not results.multi_hand_world_landmarks:
#             continue
#         for hand_world_landmarks in results.multi_hand_world_landmarks:
#             mp_drawing.plot_landmarks(
#                 hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)




