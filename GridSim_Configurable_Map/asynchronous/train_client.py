from keras.models import Model
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Input, concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import Nadam
import csv
from PIL import Image
import numpy as np

# TEST MODEL FOR ASYNC SERVER-CLIENT
# IMAGE AND TRAJECTORY TESTED USING THIS MODEL


def create_model():
    # DEFINE MODEL USING THE FUNCTIONAL API
    # FIRST THE IMAGE OF THE SIMULATOR

    cnn_activation = "relu"
    cnn_in = Input(shape=(445, 500, 3))
    model_cnn_stack = Conv2D(32, (3, 3), name="convolution0", padding="same", activation=cnn_activation)(cnn_in)
    model_cnn_stack = MaxPooling2D(pool_size=(2, 2), name="max_pool0")(model_cnn_stack)
    model_cnn_stack = Conv2D(64, (3, 3), name="convolution1", padding="same", activation=cnn_activation)(model_cnn_stack)
    model_cnn_stack = MaxPooling2D(pool_size=(2, 2), name="max_pool1")(model_cnn_stack)
    model_cnn_stack = Conv2D(64, (3, 3), name="convolution2", padding="same", activation=cnn_activation)(model_cnn_stack)
    model_cnn_stack = MaxPooling2D(pool_size=(2, 2), name="max_pool2")(model_cnn_stack)
    model_cnn_stack = Flatten()(model_cnn_stack)
    model_cnn_stack = Dropout(0.2)(model_cnn_stack)

    # NEXT STATE_INPUT
    state_in = Input(shape=(4,))
    merged = concatenate([model_cnn_stack, state_in])

    merged = Dense(64, activation=cnn_activation, name='dense0')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(10, activation=cnn_activation, name='dense1')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1, name='output')(merged)

    adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Model(inputs=[cnn_in, state_in], outputs=merged)
    model.compile(optimizer=adam, loss='mse')
    model.summary()
    return model


def read_run(fname, desired_data):
    with open(fname) as refCsv:
        data = {}
        reader = csv.DictReader(refCsv, delimiter='\t')
        for row in reader:
            for header, value in row.items():
                try:
                    if header in desired_data:
                        data[header].append(value)
                except KeyError:
                    data[header] = [value]
        return data


def actions_code(_actions):
    if _actions == ['up']:
        return 0
    elif _actions == ['down']:
        return 1
    elif _actions == ['left']:
        return 2
    elif _actions == ['right']:
        return 3
    elif _actions == ['up', 'left']:
        return 4
    elif _actions == ['up', 'right']:
        return 5
    elif _actions == ['down', 'left']:
        return 6
    elif _actions == ['down', 'right']:
        return 7
    else:
        return -1


# record data and store it in resources/recorded_data + run_nr first
# create as many runs as you want then add them here
run_list = ['run1', 'run2', 'run3', 'run4', 'run5']
model_to_train = create_model()

_image = []
_state = []
_label = []

for run in run_list:
    # model_to_train = create_model()
    dataset = read_run('../resources/recorded_data/' + run + '/state_buf.txt', ['CarPositionX', 'CarPositionY', 'Action',
                                                                                'CarAngle', 'Velocity', 'ImageName'])
    image_folder = '../resources/recorded_data/' + run + '/images/'

    print('Preparing data for: ', run, '...')
    for i in range(len(dataset['CarPositionX'])):
        current_input = []
        for header in dataset:
            current_input.append(dataset[header][i])

        repl = {"[": "", "]": "", "'": "", " ": ""}
        for x, y in repl.items():
            current_input[4] = current_input[4].replace(x, y)
        if current_input[4] == '':
            pass
        else:
            # DEFINE INPUT BATCH
            car_pos_x = float(current_input[0])
            car_pos_y = float(current_input[1])
            car_angle = float(current_input[2])
            car_velocity = float(current_input[3])
            actions = current_input[4].split(',')
            car_image_name = current_input[5]

            # CREATE ACTION CODE
            action_code = np.array(actions_code(actions))

            _input = np.array([car_pos_x, car_pos_y, car_angle, car_velocity])

            # OPEN IMAGE
            car_image = Image.open(image_folder + car_image_name)
            b, g, r = car_image.split()
            car_image = Image.merge("RGB", (r, g, b))
            _car_image = np.array(car_image)

            # CREATE BATCH
            _state.append(_input)
            _image.append(_car_image)
            _label.append(action_code)

print('Finished preparing...')
print('Starting training...')
_state = np.array(_state)
_image = np.array(_image)
_label = np.array(_label)

# FEED DATASET
plateau_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.0001, verbose=1)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
checkpoint_callback = ModelCheckpoint('../resources/recorded_data/model_checkpoint.h5', save_best_only=True, verbose=1)
callbacks = [plateau_callback, early_stopping_callback, checkpoint_callback]

model_to_train.fit(x=[_image, _state], y=_label, epochs=50, batch_size=32, validation_split=0.2, shuffle=True,
                   callbacks=callbacks)
model_to_train.save('../resources/recorded_data/model_trained.h5')
