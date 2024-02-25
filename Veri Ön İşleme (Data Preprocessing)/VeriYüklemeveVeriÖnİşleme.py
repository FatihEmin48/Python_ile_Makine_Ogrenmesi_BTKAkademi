# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:00:36 2024

@author: fatih
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme
veriler = pd.read_csv('veriler.csv')


#veri ön işleme
boy = veriler[['boy']]

boykilo = veriler[['boy','kilo']]

print(boykilo)


#Class hatırlatma
print("------ Class hatırlatma ------")
class insan:
    boy = 180
    def kosmak(self, b):
        return b+10
ali = insan()
print(ali.boy)
print(ali.kosmak(20))

l = [1,2,3]#liste
