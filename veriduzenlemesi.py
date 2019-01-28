# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:30:59 2019

@author: Hazal
"""


import cv2
import numpy as np

#Resimleri dosyadan almak için fonksiyon tanımladık.

def klasordenresimal(dosyaadi):
    resim = cv2.imread(dosyaadi)#Resmi Opencv den çektik.
    return resim

    #Verisetim 30 tane rsimden oluşuyor. Bunların hepsini almak için bir matris tanımladım.
girisverisi = np.array([])
   
for i in range(30):
       #Döngü başına geldikçe klasordenalinmisresim sıfırlansın.
       klasordenalinmisresim=0
       #Python da 0 dan başladığı için bizim setimizde 1 den başladığı için bir arttırıp başlasın.
       i=i+1
       #resimleri alıcağımız yolu gösterdik.
       string = 'veriseti/%s.jpg'%i
       klasordenalinmisresim = klasordenresimal(string)
       #Klasörden alınan resmi boyutlandırdık.
       boyutlandirilmisresim = cv2.resize(klasordenalinmisresim,(224,224))
       #Boyutlandırılmış resmi numpy fonksiyonunu klllanarak girisverisi matrisine aldım.
       print(girisverisi)
       girisverisi = np.append(girisverisi,boyutlandirilmisresim)
       
       print(i+1)
       #reshape komutu ile bize dizi kaç boyutlu olduğunu onu veriyor.
       girisverisi = np.reshape(girisverisi,(-1,224,224,3))
       #Sonra dosyadan okunup alınan resimler kendi girisverimiz adlı datasetimize kaydettik. Bu sayede sürekli gidip okumaktan kurtulduk.
       np.save("girisverimiz",girisverisi)
       
       print(girisverisi.shape)
