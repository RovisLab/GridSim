import cv2
import numpy as np


def resize_image(image, scale=0.5):

    height, width = image.shape[:2]
    res = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)
    return res


def text_on_image(image, text, choosen_font):

    cv2.putText(image, 'Neural net activations: ' + text, (0, 410), choosen_font, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA)


def init_activations_display_window(win_name, width, height, scale):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, int(width * scale), int(height * scale))


def print_activations(act_model, l_names, desired_layer_name, image_name=None, path=None, save_activations=False):

    images_per_row, images_per_col = 4, 4

    for layer_name, layer_activation in zip(l_names, act_model):
        if layer_name == desired_layer_name:
            size = layer_activation.shape  # get the shape of the activations

            display_grid = np.zeros((size[2] * images_per_col, size[1] * images_per_row), np.uint8)  # create grid
            col = 0
            row = 0

            for channel in range(0, size[3]):

                # get image from layer
                image = layer_activation[0, :, :, channel]
                image = np.rot90(image, axes=(-1, -2))
                image = np.fliplr(image)
                # print(image.shape)
                # plt.figure()
                # plt.imshow(image)
                # plt.show()
                # quit()

                # take image to grid
                display_grid[row * size[2]: (row + 1) * size[2], col * size[1]: (col + 1) * size[1]] = image

                col += 1
                if col == images_per_col:
                    row += 1
                    col = 0

            # set grid to heat map
            display_grid = cv2.applyColorMap(display_grid, cv2.COLORMAP_JET)

            # resize image and display window
            display_grid = resize_image(display_grid)

            if save_activations is True:
                if image_name is not None and path is not None:
                    cv2.imwrite(path + '/' + image_name, display_grid)
                else:
                    print('image_name and path cannot be None')
                    quit()

            # print car commands:
            # text_on_image(display_grid, car_commands, font)

            cv2.imshow(desired_layer_name, display_grid)
            cv2.waitKey(1)



