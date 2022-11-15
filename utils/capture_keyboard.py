"""
Monitor keyboard. Recommended to call "listen_keyboard" in a thread

#Example:
from threading import Thread
from captura_teclado import CapturaTeclado


# Create thread for listen keyboard
cap_keyboard = CaptureKeyboard()
threads = [Thread(target=cap_keyboard.listen_keyboard)]
for thread in threads:
    thread.start()
"""
from pynput.keyboard import Listener


class CaptureKeyboard:
    """Monitor some keys of the keyboard"""

    def __init__(self):
        self.key = 0
        self.is_screenshot = False
        self.is_released_key = False
        self.quit = False

    def clean_key(self):
        """Back to default key"""
        if self.is_released_key:
            self.key = 0

    def listen_keyboard(self):
        """Listen keys"""
        with Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _on_press(self, key):
        """Get key pressed"""
        # Forward
        if str(key) == "'w'":
            self.key = 2
            self.is_released_key = False
        # Back
        elif str(key) == "'s'":
            self.key = 5
            self.is_released_key = False
        # Left
        elif str(key) == "'a'":
            self.key = 6
            self.is_released_key = False
        # Right
        elif str(key) == "'d'":
            self.key = 3
            self.is_released_key = False
        # Up
        elif str(key) == "'q'":
            self.key = 7
            self.is_released_key = False
        # Down
        elif str(key) == "'e'":
            self.key = 4
            self.is_released_key = False
        # Lower speed
        elif str(key) == "'l'":
            self.key = 10
            self.is_released_key = False
        # Higher speed
        elif str(key) == "'h'":
            self.key = 11
            self.is_released_key = False
        # Reset environment
        elif str(key) == "'r'":
            self.key = 40
            self.is_released_key = False
        # Stop listener
        elif str(key) == 'Key.esc':
            self.key = 42
            # Stop listener
            return False

        # Pick apple
        elif str(key) == "'u'":
            self.key = 888
            self.is_released_key = False
        # Release apple
        elif str(key) == "'i'":
            self.key = 999
            self.is_released_key = False


    def _on_release(self, key):
        """Released key"""
        if str(key) == "'p'":
            self.is_screenshot = True
        else:
            self.is_released_key = True
            self.clean_key()
