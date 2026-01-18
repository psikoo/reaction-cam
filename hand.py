def count_fingers(hand_result):
    number_fingers = 0
    for hand_landmarks in hand_result.hand_landmarks:
        #              Index, Middle, Ring, Pinky
        tip_ids =     [8,     12,     16,   20   ]
        knuckle_ids = [6,     10,     14,   18   ]
        # Thumb L
        if hand_landmarks[2].x > hand_landmarks[17].x: 
            if hand_landmarks[4].x > hand_landmarks[3].x: number_fingers += 1
        # Thumb R
        else: 
            if hand_landmarks[4].x < hand_landmarks[3].x: number_fingers += 1
        # Fingers
        for i in range(4):
            if hand_landmarks[tip_ids[i]].y < hand_landmarks[knuckle_ids[i]].y: number_fingers += 1
    return number_fingers