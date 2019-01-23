import csv
import numpy as np
import pygame
from PIL import Image
import os


def read_replay_coords(fname):
    with open(fname) as csvFile:
        car_positions = []
        readCSV = csv.reader(csvFile, delimiter=',')
        for row in readCSV:
            car_position = [float(v) for v in row]
            car_positions.append(car_position)
        return car_positions


def write_replay_data(fname, *args):
    marker = check_valid_csv(fname)
    if marker == 'not_empty':
        raise OSError(fname + ' is not empty. Overwrite avoided.')

    with open(fname, "a") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        export_list = np.array([])
        for arg in args:
            export_list = np.append(export_list, arg)
        export_list = list(export_list)
        writer.writerow(export_list)


def check_valid_csv(fname):

    try:
        split_path = fname.rsplit('/', 1)[0]
        os.makedirs(split_path)
    except OSError:
        pass

    try:
        open(fname, 'x')
    except FileExistsError:
        pass

    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        try:
            first_line = next(reader)
            return 'not_empty'
        except StopIteration:
            return 'empty'


def write_state_buf(fname, args):
    fieldnames = ['CarPositionX', 'CarPositionY', 'CarAngle', 'Acceleration', 'Velocity']
    flag = check_valid_csv(fname)
    if flag == 'empty':
        try:
            with open(fname, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
                writer.writeheader()
        except :
            pass

    elif flag == 'not_empty':
        try:
            with open(fname, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
                writer.writerow({'CarPositionX': args[0],
                                 'CarPositionY': args[1],
                                 'CarAngle': args[2],
                                 'Acceleration': args[3],
                                 'Velocity': args[4]})
                                 #'ImageName': args[5]})
        except:
            pass


def save_frame(screen, frame_name, path):

    try:
        os.makedirs(path)
    except OSError:
        pass

    screendata = pygame.surfarray.array3d(screen)
    screendata = np.rot90(screendata, axes=(0, 1))
    screendata = np.flipud(screendata)
    img = Image.fromarray(screendata)
    img.save(path + '/' + frame_name)
