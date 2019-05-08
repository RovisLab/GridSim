import socket
from PIL import Image
from keras.models import load_model
import numpy as np
import tensorflow as tf

global graph, model
graph = tf.get_default_graph()
model = load_model("model_trained.h5")

# DO PREDICTION HERE


def img_client(ip, port, img_bytes, state_bytes, img_size):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        img = Image.frombytes('RGB', img_size, img_bytes)
        img_buf = np.array(img).reshape((1, img_size[1], img_size[0], 3))

        state_string = str(state_bytes, 'ascii').split(',')
        state_buf = np.array([float(x) for x in state_string]).reshape((1, 4))

        with graph.as_default():
            prediction = model.predict([img_buf, state_buf])
        sock.sendall(bytes(str(int(prediction)), 'ascii'))
        sock.close()
    finally:
        sock.close()
