from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import time

viz = Visdom()
assert viz.check_connection()

try:
    import matplotlib.pyplot as plt

except BaseException as err:
    print('Skipped matplotlib example')
    print('Error message: ', err)

"""-------Initial Setup-------"""
x, y = 0, 0
Recon_Loss = viz.line(
    X=np.array([x]),
    Y=np.array([y]),
)

Con_Loss = viz.line(
        Y=np.array([y]),
        X=np.array([x]),
)

Gen_Loss = viz.line(
        Y=np.array([y]),
        X=np.array([x]),
)

Disc_Loss = viz.line(
        Y=np.array([y]),
        X=np.array([x]),
)

the_text = viz.text("")

The_Image = viz.images(torch.zeros((8,3,512,256)),nrow=8)

def Visualization(recon_loss=0.0,gen_loss=0.0,con_loss=0.0,disc_loss=0.0,Count=0,text=None,image=None,Type=0):
    global the_acc
    global the_loss
    global the_text
    global the_image
    global Recon_Loss

    if(Type == 0):
        viz.line(
            X=np.array([Count]),
            Y=np.array([recon_loss]),
            win=Recon_Loss,
            update='append',
            opts=dict(
                title='Recon_loss',
                linecolor=np.array([[255, 0, 255]])
            )
        )
        time.sleep(0.00001)

        viz.images(
            image,
            win=The_Image,
            nrow=8,
        )

        viz.text(text,
                 win=the_text,
                 append=True)
    else:
        viz.line(
            X=np.array([Count]),
            Y=np.array([gen_loss]),
            win=Gen_Loss,
            name='Gen',
            update='append',
            opts=dict(
                title='Gen_loss',
                linecolor=np.array([[224,17,93]])
            )
        )

        viz.line(
            X=np.array([Count]),
            Y=np.array([con_loss]),
            win=Con_Loss,
            name='Con',
            update='append',
            opts=dict(
                title='Con_loss',
                linecolor=np.array([[24,173,142]])
            )
        )

        viz.line(
            X=np.array([Count]),
            Y=np.array([disc_loss]),
            win=Disc_Loss,
            name='Disc',
            update='append',
            opts=dict(
                title='Disc_loss',
                linecolor=np.array([[28, 191, 28]])
            )
        )

        viz.images(
            image,
            win=The_Image,
            nrow=8,
        )

        viz.text(text,
                 win=the_text,
                 append=True)