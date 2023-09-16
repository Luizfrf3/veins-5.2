# -*- coding: utf-8 -*-
# Calculate communication range in meters

import os
import sys
import math

T = float(input('MinPowerLevel (dBm): '))
P = float(input('TxPower (mW): '))

d = 0.00404913 * pow(10, (-T/20)) * math.sqrt(P)

print('Range (m):',"{:.2f}".format(d))
