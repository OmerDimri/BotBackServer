import os

PRODUCTION = "production"
SMALLEST_INTERVAL = ""
BIGGEST_INTERVAL = ""
FEE_RATE = 0.00027

def set_smallest_biggest_interval(smallest, biggest):
    global SMALLEST_INTERVAL, BIGGEST_INTERVAL
    SMALLEST_INTERVAL = smallest
    BIGGEST_INTERVAL = biggest


def get_smallest_biggest_interval():
    return SMALLEST_INTERVAL, BIGGEST_INTERVAL
