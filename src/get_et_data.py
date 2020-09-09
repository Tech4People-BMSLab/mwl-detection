"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-05-07 12:23:47
@Last Modified by:   Tenzing Dolmans
@Description: Finds all connected eye trackers, makes an LSL stream
when "Link" is selected in the GUI.
"""
import tobii_research as tr
from pylsl import StreamInfo, StreamOutlet
from tkinter import Frame, Button, Tk


def find_et():
    """Make a list of all eyetrackers connected to your PC,
    return the first one."""
    found_eyetrackers = tr.find_all_eyetrackers()
    eyetracker = found_eyetrackers[0]
    return eyetracker


def make_stream(eyetracker):
    """Creates a stream outlet and defines meta-data.
    https://labstreaminglayer.readthedocs.io/"""
    info = StreamInfo('TobiiET', 'ET', 1, 120, 'float32',
                      eyetracker.serial_number)
    info.desc().append_child_value("manufacturer", "Tobii")
    channels = info.desc().append_child("channels")
    for c in ["LX", "LY", "RX", "RY"]:
        channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "normalised location") \
            .append_child_value("type", "ET")
    return StreamOutlet(info)


def gaze_data_callback(gaze_data):
    """This is the callback function used in the "subscribe_to".
    Note that I am unpacking the gaze_data into several parts that have
    type float_32."""
    gaze_data_left_x = gaze_data['left_gaze_point_on_display_area'][0]
    gaze_data_left_y = gaze_data['left_gaze_point_on_display_area'][1]
    gaze_data_right_x = gaze_data['right_gaze_point_on_display_area'][0]
    gaze_data_right_y = gaze_data['right_gaze_point_on_display_area'][1]
    chunk = [gaze_data_left_x, gaze_data_left_y,
             gaze_data_right_x, gaze_data_right_y]
    outlet.push_chunk(chunk)


class Window(Frame):
    """Simple GUI that contains buttons for the linking and unlinking
    of the Tobii stream."""
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.et = find_et()
        self.pack(fill=BOTH, expand=1)
        startButton = Button(self, text="Link",
                             command=self.clickStartButton)
        startButton.place(x=50, y=80)
        stopButton = Button(self, text="Unlink",
                            command=self.clickStopButton)
        stopButton.place(x=200, y=80)

    def clickStartButton(self):
        print("Subscribing to time synchronization data.")
        self.et.subscribe_to(tr.EYETRACKER_GAZE_DATA,
                             gaze_data_callback,
                             as_dictionary=True)

    def clickStopButton(self):
        self.et.unsubscribe_from(tr.EYETRACKER_GAZE_DATA,
                                 gaze_data_callback)
        print("Unsubscribed from time synchronization data.")


if __name__ == "__main__":
    eyetracker = find_et()
    outlet = make_stream(eyetracker)
    root = Tk()
    app = Window(root)
    root.wm_title("ET Data Streamer")
    root.geometry("300x200")
    root.mainloop()
