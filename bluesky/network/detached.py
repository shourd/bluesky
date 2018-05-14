""" Node encapsulates the sim process, and manages process I/O. """
import os
import bluesky
from bluesky.tools import Timer


class Node(object):
    def __init__(self):
        self.node_id = b'\x00' + os.urandom(4)
        self.host_id = b''
        self.running = True

        # Tell bluesky that this client will manage the network I/O
        bluesky.net = self

    def event(self, eventname, eventdata, sender_id):
        ''' Event data handler. Reimplemented in Simulation. '''
        print('Node {} received {} data from {}'.format(self.node_id, eventname, sender_id))

    def step(self):
        ''' Perform one iteration step. Reimplemented in Simulation. '''
        pass

    def start(self):
        ''' Starting of main loop. '''
        print('Node started, id={}'.format(self.node_id))

        # run() implements the main loop
        self.run()

    def quit(self):
        ''' Quit the simulation process. '''
        self.running = False

    def run(self):
        ''' Start the main loop of this node. '''
        while self.running:
            # Perform a simulation step
            self.step()

            # Process timers
            Timer.update_timers()

    def addnodes(self, count=1):
        pass

    def send_event(self, eventname, data=None, target=None):
        pass

    def send_stream(self, name, data):
        pass
