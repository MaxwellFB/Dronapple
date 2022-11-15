"""
Control the drone using keyboard.
This class was used to capture images to train the object detector and for environment tests
"""
import os
import time
from threading import Thread

import gym
import cv2

from utils.capture_keyboard import CaptureKeyboard


speed = 0.5

environment = gym.make(
    "airgym:airsim-drone-sample-v0",
    ip_address="127.0.0.1",
    step_length=speed,
    nstep_limit=200000,
    image_shape=(256, 144, 3),
    environment=1,
    threshold_spawn_apple=40
)
environment.disable_percentage_apple()
environment.set_auto_grab(False)
environment.reset()

# Start keyboard listener
cap_keyboard = CaptureKeyboard()
threads = [Thread(target=cap_keyboard.listen_keyboard)]
for thread in threads:
    thread.start()

step = len(os.listdir('data/images_control_keyboard'))
while True:
    test = 0
    # Look at the file "capture_keyboard" to see the keys used to move the drone and do actions
    actions = cap_keyboard.key

    # Get apple
    if actions == 888:
        actions = 0
        test = 888
    # Release apple
    elif actions == 999:
        actions = 0
        test = 999
    # Lower speed
    elif actions == 10:
        speed -= 0.1
        actions = 0
        print(f'Speed: {speed}')
    # Higher speed
    elif actions == 11:
        speed += 0.1
        print(f'Speed: {speed}')
        actions = 0
    # Reset environment
    if actions == 40:
        environment.reset()
    # Exit
    elif actions == 42:
        break
    else:
        obs, reward, terminal, state, _ = environment.step(actions, step_length=speed, test=test)

        # Press "p" for screenshot
        if cap_keyboard.is_screenshot:
            cap_keyboard.is_screenshot = False
            cv2.imwrite(f'data/images_control_keyboard/{step}.png', obs)
            time.sleep(0.1)
            step += 1
