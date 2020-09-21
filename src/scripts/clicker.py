from pynput.mouse import Button, Controller
import time
import threading

mouse = Controller()


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


def clicker():

    mouse.position = (3600, 180)
    mouse.click(Button.left, 1)
    time.sleep(2)
    mouse.click(Button.left, 1)
    time.sleep(2)
    mouse.click(Button.left, 1)
    time.sleep(2)
    # mouse.position = (2650, 190)
    # mouse.click(Button.left, 1)
    # time.sleep(2)
    # mouse.click(Button.left, 1)
    # time.sleep(2)
    # mouse.click(Button.left, 1)
    # time.sleep(2)
    mouse.position = (500, 500)


def main():
    t = set_interval(clicker, 300)


if __name__ == "__main__":
    main()


def clicker():
    for i in range(50):
        mouse.position = (918, 539)
        time.sleep(0.1)
        mouse.click(Button.left, 1)
        time.sleep(0.1)
        mouse.position = (920, 560)
        time.sleep(0.1)
        mouse.click(Button.left, 1)
        time.sleep(0.1)
