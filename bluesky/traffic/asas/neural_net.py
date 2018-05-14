# -*- coding: utf-8 -*-
"""
Created on Sat May 12 2018

@author: Sjoerd van Rooijen

This resolution algorithm uses the SSD as input for a Neural Network to obtain resolutions

Limitations:
- Only 2 A/C scenarios
- Only HDG resolutions
- Binary resolutions (left/right)

"""
import numpy as np
from keras.models import model_from_json
import os
# from bluesky.tools.aero import ft,fpm,kts
# from math import ceil


def start(asas):
    print('Resolution algorithm start')
    global NN_model
    NN_model = load_model('first_model')
    print('NN model loaded')

def resolve(asas, traf):
    """ Resolve all current conflicts """
    # SETTINGS

    reso_hdg = 25

    # --------------

    if 'ssd_image_global' not in globals():  # only execute if ssd is created
        print('SSD not defined yet')
        return

    print('Resolve')

    # Check if ASAS is ON first!
    if not asas.swasas:
        return

    # Stores resolution vector, also used in visualization
    asas.asasn        = np.zeros(traf.ntraf, dtype=np.float32)
    asas.asase        = np.zeros(traf.ntraf, dtype=np.float32)

    prediction, certainty = predict_resolution(ssd_image_global)

    # The old speed vector, cartesian coordinates
    v = np.array([traf.gseast, traf.gsnorth, traf.vs])
    # the old heading converted to degrees
    hdg = (np.arctan2(v[0, 0], v[1, 0]) * 180 / np.pi) % 360

    if prediction == 0: # left
        newtrack = (hdg - reso_hdg + 360) % 360 # add 360 in case hdg becomes negative
    elif prediction == 1: # right
        newtrack = (hdg + reso_hdg) % 360
    else:
        print('Error: Unexpected prediction')
        newtrack = hdg

    # Get indices of aircraft that have a resolution
    ids = 0 # old: dv[0,:] ** 2 + dv[1,:] ** 2 > 0

    # Now assign resolutions to variables in the ASAS class
    asas.trk[0] = newtrack

    # Stores resolution vector
    asas.asase[ids] = asas.tas[ids] * np.sin(asas.trk[ids] / 180 * np.pi)
    asas.asasn[ids] = asas.tas[ids] * np.cos(asas.trk[ids] / 180 * np.pi)

    # asaseval should be set to True now
    if not asas.asaseval:
        asas.asaseval = True

    # If resolutions are limited in the horizontal direction, then asasalt should
    # be equal to auto pilot alt (aalt).
    # asas.alt = asas.aalt


#======================= NN Predictor ===========================

def predict_resolution(ssd_image, size=(120,120)):
    """ This function predicts a resolution based on a downsampled SSD image """

    imgdata = np.array(ssd_image)[:,:,1] # convert PIL object to array
    imgdata = imgdata.reshape(1, size[0], size[1], 1) # reshape to 4 dimensions
    probabilities = NN_model.predict(imgdata) # evaluate model
    print(probabilities)

    prediction = int(np.argmax(probabilities)) # obtain predicted value
    certainty = int(round(probabilities[0][prediction], 2) * 100) # obtain probability of prediction

    classes = ['Left','Right']

    print('Advisory: {} with {}% certainty'.format(classes[prediction], certainty))

    return prediction, certainty

#=================== Auxiliary functions =========================

def load_model(model_name='first_model'):
    """ Load JSON model from disk """
    print("Start loading model.")
    a = os.path.abspath(__file__)
    model_path = 'models/' + model_name
    try:
        json_file = open('{}.json'.format(model_path), 'r')
    except FileNotFoundError:
        print('Model not found')
        return

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('{}.h5'.format(model_path))

    return loaded_model
