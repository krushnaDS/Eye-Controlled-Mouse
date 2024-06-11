import cv2
import mediapipe as mp
import pyautogui

##  creates a video capture object (cam) that accesses your computer's default webcam (index 0).
cam = cv2.VideoCapture(0)

## This line creates a MediaPipe FaceMesh object (face_mesh) for detecting facial landmarks with higher precision.
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)

## gets the width and height of your computer screen and stores them in variables screen_w and screen_h.
screen_w,screen_h = pyautogui.size()


while True:
    ## This line reads a single frame from the webcam and stores it in the frame variable.
    # The underscore (_) discards the first returned value (a status flag) as we only need the frame itself.
    _,frame = cam.read()

    ## This line flips the frame horizontally (mirrors it) because webcams often show a reversed image.
    frame = cv2.flip(frame,1)

    ## This line converts the frame from BGR color format (used by OpenCV) to RGB format (used by MediaPipe) for face mesh detection.
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    ## This line processes the frame using the face mesh model and stores the results in the output variable.
    output = face_mesh.process(rgb_frame)

    ## This line extracts the detected facial landmarks (if any) from the output and stores them in landmark_points.
    landmark_points = output.multi_face_landmarks

    ## This line extracts the height and width of the frame and discards an unused value.
    frame_h,frame_w,_ = frame.shape

## This line checks if any faces were detected in the frame.
    if landmark_points:
        landmarks = landmark_points[0].landmark

        ## This loop iterates over landmarks between index 474 and 477, which correspond to the right eye
        for id,landmark in enumerate(landmarks[474:478]):
            ## This line calculates the horizontal position (x) of the landmark on the frame by multiplying its normalized x-coordinate with the frame width.
            x = int(landmark.x * frame_w)

            ## This line calculates the vertical position (y) of the landmark on the frame by multiplying its normalized y-coordinate with the frame height.
            y = int(landmark.y * frame_h)

            ## This line draws a green circle (radius 3) at the calculated landmark position on the frame for visualization.
            cv2.circle(frame,(x,y),3,(0,255,0))

            ## This block controls the mouse movement using the right eye's first landmark (index 1).
            if id == 1:

                ## This line calculates the corresponding x-coordinate on the screen based on the normalized x-coordinate of the landmark and the screen width.
                screen_x = int(landmark.x * screen_w)

                ## This line calculates the corresponding y-coordinate on the screen based on the normalized y-coordinate of the landmark and the screen height.
                screen_y = int(landmark.y*screen_h)

                ## This line uses pyautogui to move the mouse cursor to the calculated screen position.
                pyautogui.moveTo(screen_x,screen_y)

        ## This line creates a list named left containing the two relevant landmarks for the left eye
        left = [landmarks[145],landmarks[159]]

        ## This section focuses on detecting a potential left eye blink to trigger a mouse click.

        ## This loop iterates through the two landmarks stored in the left list (presumably representing the top and bottom eyelid of the left eye).
        for landmark in left:
            ## This line calculates the horizontal position (x) of the current landmark on the frame, similar to the previous loop for the right eye.
            x = int(landmark.x * frame_w)
            ## This line calculates the vertical position (y) of the current landmark on the frame
            y = int(landmark.y * frame_h)

            ## This line draws a blue circle (radius 3) at the calculated landmark position for visualization, helping you see the movement of the eyelids.
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        ## This is the core condition for detecting a potential blink.
        # It compares the vertical distance between the top and bottom eyelid landmarks stored in left.
        # The threshold value is 0.009, which is a very small number. This means that if the difference in y-coordinates (distance) between the two landmarks is less than 0.009,
        # it signifies the eyelids are close together, potentially indicating a blink.
        if (left[0].y - left[1].y) < 0.009:

            # If the condition is true (likely a blink), this line triggers a mouse click using pyautogui.
            pyautogui.click()
            # This line pauses the program for 1 second after the click to prevent rapid clicking due to small eyelid movements.
            pyautogui.sleep(1)

    ## This line displays the processed frame containing the circles around the landmarks and the mouse cursor on your screen. The window title is "Eye Controlled Mouse".
    cv2.imshow("Eye Controlled Mouse",frame)

    # This line waits for a key press for 1 millisecond. In this program, it essentially keeps the program running and displaying the video stream. Pressing any key will close the program.

    cv2.waitKey(1)