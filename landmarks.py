import cv2
import numpy

def draw_landmarks_on_image_hand(DRAW_HAND_BOX, DRAW_HAND_NUMBERS, DRAW_HAND_CIRCLES, rgb_image, hand_result, number_fingers):
    if not hand_result or not hand_result.hand_landmarks: return rgb_image
    annotated_image = numpy.copy(rgb_image)
    height, width, _ = annotated_image.shape

    CONNECTIONS = [
        (0 , 1 ), (1 , 2 ), (2 , 3 ), (3 , 4 ),  # Thumb
        (0 , 5 ), (5 , 6 ), (6 , 7 ), (7 , 8 ),  # Index
        (9 , 10), (10, 11), (11, 12),            # Middle
        (13, 14), (14, 15), (15, 16),            # Ring
        (0 , 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5 , 9 ), (9 , 13), (13, 17)             # Palm
    ]

    for hand_landmarks in hand_result.hand_landmarks: 
        minX, minY, _ = annotated_image.shape
        maxX = 0 
        maxY = 0
        # Draw input
        for index, landmark in enumerate(hand_landmarks):
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            if px < minX: minX = px
            if py < minY: minY = py
            if px > maxX: maxX = px
            if py > maxY: maxY = py
            if DRAW_HAND_NUMBERS:
                cv2.putText(annotated_image, text=str(index), org=(px, py), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            if DRAW_HAND_CIRCLES:
                cv2.circle(annotated_image, (px, py), 5, (0, 0, 255), -1)
        # Draw connections
        for connection in CONNECTIONS:
            start = (int(hand_landmarks[connection[0]].x * width), int(hand_landmarks[connection[0]].y * height))
            end   = (int(hand_landmarks[connection[1]].x * width), int(hand_landmarks[connection[1]].y * height))
            cv2.line(annotated_image, start, end, (255, 255, 255), 2)
        # Draw bounds
        if DRAW_HAND_BOX:
            BOUNDS = [ ((minX, minY), (minX, maxY)), ((minX, maxY), (maxX, maxY)), ((maxX, maxY), (maxX, minY)), ((maxX, minY), (minX, minY)) ]
            for bound in BOUNDS:
                cv2.line(annotated_image, bound[0], bound[1], (0, 0, 0), 2)
        # Draw hand
        if hand_landmarks[2].x > hand_landmarks[17].x: 
            cv2.putText(annotated_image, text="L", org=(minX+3, maxY-3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
        else: 
            cv2.putText(annotated_image, text="R", org=(minX+3, maxY-3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
        # Draw number of finger
        cv2.putText(annotated_image, text=str(number_fingers), org=(0+3, height-3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

    return annotated_image

def draw_landmarks_on_image_face(DRAW_FACE_BOX, DRAW_FACE_NUMBERS, DRAW_FACE_CIRCLES, rgb_image, face_result):
    if not face_result or not face_result.face_landmarks: return rgb_image
    annotated_image = numpy.copy(rgb_image)
    height, width, _ = annotated_image.shape

    for face_landmarks in face_result.face_landmarks: 
        minX, minY, _ = annotated_image.shape
        maxX = 0 
        maxY = 0
        # Draw input
        for index, landmark in enumerate(face_landmarks):
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            if px < minX: minX = px
            if py < minY: minY = py
            if px > maxX: maxX = px
            if py > maxY: maxY = py
            if DRAW_FACE_NUMBERS: 
                cv2.putText(annotated_image, text=str(index), org=(px, py), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.1, color=(0, 0, 255), thickness=2)
            if DRAW_FACE_CIRCLES: 
                cv2.circle(annotated_image, (px, py), 2, (0, 0, 255), -1)
        # Draw bounds
        if DRAW_FACE_BOX:
            BOUNDS = [ ((minX, minY), (minX, maxY)), ((minX, maxY), (maxX, maxY)), ((maxX, maxY), (maxX, minY)), ((maxX, minY), (minX, minY)) ]
            for bound in BOUNDS:
                cv2.line(annotated_image, bound[0], bound[1], (0, 0, 0), 2)

    return annotated_image