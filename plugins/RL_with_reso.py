"""
This Plugin is created to give a resolution advisory based on a neural network
"""

from bluesky import stack, sim, traf  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.aero import kts
from bluesky.traffic.asas import ASAS
from bluesky.traffic.asas.SSD import initializeSSD, constructSSD
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from os import path, makedirs
import shutil # to remove folder
import itertools # calculates cartesian product
from keras.models import model_from_json


### Do not change name of the following function:
def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':      'RL_with_reso',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':      'sim',

        # Update interval in seconds.
        'update_interval':  10,

        # The update function is called after traffic is updated.
        'update':           update,

        # The preupdate function is called before traffic is updated.
        'preupdate':        preupdate,

        'reset':            reset
        }

    stackfunctions = {
        # The command name for your function
        'RL': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'Type RL ON to start generating data',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff]',

            # The name of your function in this plugin
            myfun,

            # a longer help text of your function.
            'This plugin generates SSD-resolution data to be used for supervised learning.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Initialization function:
def myfun(flag=True):
    global RLflag
    RLflag = flag # convert flag var to global var

    if flag:                    # load scenario if plugin is turning on
        stack.stack('OP')       # start time
        stack.stack('FF')       # Fast forward to first scenario iteration
        shutil.rmtree('output') # delete output folder
        makedirs('output')      # create new output folder

    global NN_model
    NN_model = load_model('first_model')
    print('Model loaded')

    return True, 'The RL plugin is turned %s.' % ('on' if flag else 'off')


def reset():
    pass


def preupdate():
    pass
    if 'RLflag' in globals() and RLflag:  # only execute if plugin is turned on
        pass


def update():
    if 'RLflag' not in globals() or RLflag == False: # only execute if plugin is turned on
        return

    # load scenario files at appropriate times with varying init conditions
    load_scenarios()

    if not traf.asas.inconf.any(): # stop execution when there are no conflicts
        return

    # construct SSD and save downsampled image to disk
    create_SSD()

    # save resolution to txt file
    save_resolution()


def load_scenarios():
    """
    ALL SCENARIO SETTINGS

    """
    # implement: show original heading delta, and after alterarion.
    scenario_duration   =   10  # [min]
    lookahead_time      =   180  # [sec]

    start_angle         =   40  # [deg]
    stop_angle          =   320 # [deg]
    angle_increments    =   20   # [deg]

    t_cpa_scenarios     =   [270]  # range of t_cpa's
    cpa_scenarios       =   [-1.5, 1.5]  # range of cpa's
    angle_scenarios     =   np.arange(start_angle, stop_angle, angle_increments)
    total_scenarios     =   len(angle_scenarios) * len(t_cpa_scenarios) * len(cpa_scenarios)

    # start loading scenarios
    all_scenarios = list(itertools.product(angle_scenarios, cpa_scenarios, t_cpa_scenarios))

    if 'scenario_count' not in globals():
        global scenario_count
        scenario_count = 0
        global t_cpa
        t_cpa = 0

    ### if scenario is over, or if it is the first scenario or if LOS has occured, enter loop:
    if scenario_count < total_scenarios and int(sim.simt) > scenario_duration * 60 or scenario_count == 0 or traf.asas.lospairs_all:

        # check if LOS occured in previous scenario and log
        if traf.asas.lospairs_all:  # if lospairs_all contains LOS pairs:
            print('Loss of Separation!')
            with open('output/LOS_tracker.txt', "a") as text_file:
                text_file.write("LOS: S{}\n".format(scenario_count))


        scenario        = all_scenarios[scenario_count]
        conflict_angle  = scenario[0]  # generate current conflict angle
        cpa             = scenario[1]
        t_cpa           = scenario[2]

        scenario_count += 1

        stack.stack('HOLD')
        stack.stack('RESET')
        stack.stack('SCEN SCENARIO {}'.format(scenario_count))
        stack.stack('ECHO CONFLICT ANGLE: {conflict_angle} deg, CPA: {CPA} nm, T_CPA: {tCPA} s'
                    .format(conflict_angle=conflict_angle, CPA=cpa, tCPA=t_cpa))
        stack.stack('PCALL SJOERD_WITH_RESO {conflict_angle} {CPA} {tCPA}'
                    .format(conflict_angle=conflict_angle,CPA=cpa, tCPA=t_cpa))

        traf.asas.dtlookahead = lookahead_time  # set lookahead time of ASAS algorithm

    if scenario_count == total_scenarios:
        stack.stack('ECHO ALL SCENARIOS DONE')
        stack.stack('HOLD')


def create_SSD():
    # SETTINGS
    rotate_flag = False  # rotate SSDs before saving?
    show_SSD_flag = True
    size = 120, 120  # determine SSD image dimensions for saving


    ''' CONSTRUCTING SSD PARAMS '''
    asas = ASAS()

    # if True not in traf.asas.inconf:  # skip update if no a/c in conflict
    if not traf.asas.inconf[0]:  # if first a/c not in conflict. skip function
        return

    initializeSSD(asas, traf.ntraf)

    # traf.asas.mar = 1.4

    for i in range(traf.ntraf):
        asas.inconf[i] = True

    constructSSD(asas, traf, "RS1")

    # calculate_resolution(asas, Traffic, 'min')

    ''' VISUALIZING SSD'''

    # SSD - CIRCLES OF VMAX AND VMIN
    vmin = asas.vmin
    vmax = asas.vmax
    N_angle = 180

    angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
    xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
    SSD_lst = [list(map(list, np.flipud(xyc * vmax))), list(map(list, xyc * vmin))]

    # Outer circle: vmax
    SSD_outer = np.array(SSD_lst[0])
    x_SSD_outer = np.append(SSD_outer[:, 0], np.array(SSD_outer[0, 0]))
    y_SSD_outer = np.append(SSD_outer[:, 1], np.array(SSD_outer[0, 1]))
    # Inner circle: vmin
    SSD_inner = np.array(SSD_lst[1])
    x_SSD_inner = np.append(SSD_inner[:, 0], np.array(SSD_inner[0, 0]))
    y_SSD_inner = np.append(SSD_inner[:, 1], np.array(SSD_inner[0, 1]))

    idx = 0  # Take first aircraft for now. Dynamic: traf.id.index("KL1")
    v_own = np.array([traf.gseast[idx], traf.gsnorth[idx]])

    # PLOTS
    fig = plt.figure('SSD')
    plt.clf()
    # plt.ion()
    plt.plot(x_SSD_outer, y_SSD_outer, color='gray')
    plt.plot(x_SSD_inner, y_SSD_inner, color='gray')

    if asas.FRV[idx]:
        for j in range(len(asas.FRV[idx])):
            FRV_1 = np.array(asas.FRV[idx][j])
            x_FRV1 = np.append(FRV_1[:, 0], np.array(FRV_1[0, 0]))
            y_FRV1 = np.append(FRV_1[:, 1], np.array(FRV_1[0, 1]))
            plt.plot(x_FRV1, y_FRV1, '-', color='red')
            plt.fill(x_FRV1, y_FRV1, color='red')

    # plt.arrow( x, y, dx, dy, **kwargs )
    plt.arrow(0, 0, v_own[0], v_own[1], fc='limegreen', ec='limegreen', head_width=15, head_length=10,
              linewidth=2.5, zorder=10, length_includes_head=True)

    plt.axis('equal')
    plt.axis('off')

    if show_SSD_flag:  # show SSD during computation
        plt.draw()
        plt.pause(0.001)

        # convert figure to array
        ssd_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')  # one long string of bytes
        ssd_image = ssd_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # convert to array [height,width,RGB
        ssd_image = ssd_image[:, 80:560]  # make image square

        # convert to image object
        ssd_image = Image.fromarray(ssd_image)
    else:  # don't show SSD during computation (faster)
        plt.savefig('SSD.png', bbox_inches='tight')
        ssd_image = Image.open('SSD.png')

    # rotate image
    if rotate_flag:
        rotate_angle = round(traf.hdg[0]) - 360  # deviation from north # 0 denotes first a/c
        ssd_image = ssd_image.rotate(rotate_angle,
                                     resample=Image.BILINEAR)  # rotate ssd to have velocity vector north faced

        # Make all black pixels caused by rotation white
        ssd_image = ssd_image.convert("RGB")
        imgdata = ssd_image.getdata() # obtain pixel data

        newData = []
        for pixel in imgdata:
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                newData.append((255, 255, 255))
            else:
                newData.append(pixel)

        ssd_image.putdata(newData)

    # crop white space
    ssd_image = trim(ssd_image)

    # downsample image
    ssd_image_ds = ssd_image.resize(size,
                                    Image.BILINEAR)  # The filter argument can be one of NEAREST, BILINEAR, BICUBIC, ANTIALIAS (filter=)

    # save SSD to disk
    ssd_image_ds.save('output/SSD_S{scenario_count}_T{simtime}.png'
                      .format(scenario_count=scenario_count, simtime=int(sim.simt)))  # save file with S {scenario counter} T {sim time}

    predict_resolution(ssd_image_ds, size)

    """ TODO: Get ssd_image in the right format and input into: prediction = model.predict(image) then profit! """


def save_resolution():
    """ SAVE RESOLUTION TO TXT FILE """

    # SETTINGS
    bucket = 5  # discretize output resolution to nearest x [deg]

    # Make resolution relative to current heading
    res_hdg = traf.asas.trk[0]
    delta_hdg = int(res_hdg - traf.hdg[0])                   # correct for current heading
    if delta_hdg < -180: delta_hdg = delta_hdg + 360
    if delta_hdg > 180: delta_hdg = delta_hdg - 360

    # discretize to nearest 5 deg
    delta_hdg = bucket * np.round(delta_hdg/bucket)

    # SPD
    res_spd = traf.asas.tas[0] / kts
    current_spd = traf.tas[0] / kts
    delta_spd = int(res_spd - current_spd)

    # discretize to nearest 5 kts
    delta_spd = bucket * np.round(delta_spd / bucket)

    # Save resolutions to txt file
    filename = 'output/resolutions_S{}.txt'.format(scenario_count)

    if not path.isfile(filename):
        with open(filename,"w") as text_file:
            text_file.write("This file contains the relative resolution vectors (HDG; SPD) per 10s:\n")

    with open(filename, "a") as text_file:
        text_file.write("{}; {}\n".format(delta_hdg, delta_spd))


def predict_resolution(ssd_image, size):
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


""" AUXILIARY FUNCTIONS """

### Trim images
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def load_model(model_name):
    """ Load JSON model from disk """
    print("Start loading model.")
    # t = time.time()
    model_path = 'models/' + model_name
    try:
        json_file = open('{}.json'.format(model_path), 'r')
    except FileNotFoundError:
        print('Model not found')
        return

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('{}.h5'.format(model_path))
    # elapsed = round(time.time() - t, 2)
    # print("Loaded model from disk. ({} sec)".format(elapsed))

    return loaded_model