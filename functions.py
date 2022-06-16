
from binascii import crc32
import enum
from msilib.schema import Directory
import os
import time
import cv2
import json


import sys
from PIL import Image
import re
from pathlib import Path
import oauthlib
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

import shutil
import pytesseract 
from pytube import YouTube

import easyocr
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from math import sqrt 
from math import floor

###################################### FUNCIONES EXTRA ######################################
def addJ (name):
    # return "test/"+str(name)+".jpg"
    return str(name)+".jpg"

# Lista de nombres de Frames como numero
def ls(ruta = Path.cwd()):
    return [int(arch.name.split(".")[0]) for arch in Path(ruta).iterdir() if (arch.is_file() and re.search(r'\.jpg$', arch.name))]

# Lista de nombres de Frames	
def ls2(ruta = Path.cwd()):
    return [arch.name for arch in Path(ruta).iterdir() if (arch.is_file() and re.search(r'\.jpg$', arch.name))]



###################################### FUNCIONES PARA PEGAR IMAGENES SIMILARES (la use para los modelos) ######################################
def mush(array, orute ,drute, name): 
    '''
    toma un conjunto de imagenes (array) y los une en una sola imagen de nombre <name>
    -------------------------------------------------------
    array: lista con nombres de las fotos, 
    orute: ruta origen frames
    drute: ruta destino frames
    name: nombre final imagen
    '''
    images = [Image.open(orute+"/"+str(x)) for x in array]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(drute+"/"+name)
    return 1

def bloques(f_ruta, nombre ):
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))
    print("Frames totales: %d" % len(Frames))

    if (not os.path.isdir("./bloques_"+nombre)):
        os.mkdir("./bloques_"+nombre)

    # #definicion de variables
    # height, width, channels = 0, 0 , 0
    # pix_num = 0
    # anterior = 0
    # # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    # for index, i in enumerate(Frames):

    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    lista = []
    j = 0 
    for a, frame in enumerate(Frames):
        i = int(frame.split(".")[0]) 
        im = cv2.imread(f_ruta+frame)
        if (a  == 0):
            # Numero de pixeles por frame
            height, width, channels = im.shape
            pix_num = height * width * 3
            im2 = im.copy()
            img2 = im.copy()
            anterior = i
            lista.append(frame)
            continue

        img1 = img_as_float(img2)
        img2 = img_as_float(im)
        im1 = im2
        im2 = im 
        # Aplicando metrica SSIM
        ssimV = ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min())
        # total number of different pixels between im1 and im2
        dif = np.sum(im1 != im2)
        # if (False):
        if ( dif/pix_num < 0.5):
            #  Son escencial- la misma
            lista.append(str(anterior)+".jpg")
            direc = f_ruta+str(anterior)+".jpg"
            # os.remove(direc)
        elif(ssimV>0.85):			
            # Son escencial- la misma
            lista.append(str(anterior)+".jpg")
            direc = f_ruta+str(anterior)+".jpg"
            # os.remove(direc)
        else:
            if(len(lista) != 0 ):
                mush(lista, f_ruta, "./bloques_"+nombre, str(j)+'.jpg')
                j += 1
                lista = []
            else:
                lista.append(str(anterior)+".jpg")


        anterior = i

    return "OK"

def pares(f_ruta, nombre):
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))

    if (not os.path.isdir("./pares_"+nombre)):
        os.mkdir("./pares_"+nombre)

    # #definicion de variables
    # height, width, channels = 0, 0 , 0
    # pix_num = 0
    # anterior = 0
    # # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    # for index, i in enumerate(Frames):

    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    # lista = []
    j = 0 
    for a, frame in enumerate(Frames):
        i = int(frame.split(".")[0]) 
        if(a != 0):
            rute1 = f_ruta+ str(anterior)+'.jpg'
            rute2 = f_ruta+ str(i)+'.jpg'
            if(not isame(rute1, rute2)):
                print("Frame: %d" % j)
                lista = []
                lista.append(str(anterior)+".jpg")
                lista.append(str(i)+".jpg")
                mush(lista, f_ruta, "./pares_"+nombre, str(j)+'.jpg')
                j += 1

        anterior = i

    return "OK"



###################################### FUNCIONES PRINCIPALES ######################################
def download_video(url):
    '''
    CAMBIOS EN CIPHER.PY
    lineas 272 y 273
    r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
    r'\([a-z]\s*=\s*([a-zA-Z0-9$]{2,3})(\[\d+\])?\([a-z]\)'
    cambiar linea  288
    nfunc=re.escape(function_match.group(1))),
    '''
    # for stream in video.streams:
    #     print(stream)
    video = YouTube(url)
    video = video.streams.get_highest_resolution()
    video.download()

def getFrames(ruta, saltos, escala = 100 ,fname = "Default"):
    '''
    Crea una carpeta con el nombre del video e inserta los frames en esta con nombres 0,1,2,3,...
    -------------------------------------------------------
    ruta: ruta origen del video
    saltos: saltos de  fps veces
    escala: tamano final de la imagen (en porcentaje)
    '''

    inicio = time.time()
    VideoName = ruta.split("/")[-1]
    RutaVideo = ruta.replace(VideoName, '')
    print("----------------------------------------")
    print("VideoName : |"+VideoName+"|")
    print("RutaVideo : |"+RutaVideo+"|")

    VideoName2 = VideoName.replace("y2mate.com", "").replace(".mp4", "").replace(" ", "").replace(".", "").replace("-", "")
    rutaFrames = RutaVideo+"F_"+VideoName2
    if (not os.path.isdir(rutaFrames)):
        os.mkdir(rutaFrames)

    # ------- Obteniendo/guardando la cantidad minima de Frames sin perder info -------
    vidcap = cv2.VideoCapture(RutaVideo+VideoName)
    VideoName = VideoName.replace("y2mate.com", "").replace(".mp4", "").replace(" ", "").replace(".", "").replace("-", "")
    # RutaVideo = RutaVideo+"Frames/"
    RutaVideo = RutaVideo+"F_"+VideoName+"/"

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = int(vidcap.get(cv2.CAP_PROP_FPS))
    print("FPS -> %d " % fps)
    print("Numero de frames -> %d " % length)
    print("Se selecciona 1 frame cada %d " % fps*saltos)

    count = 0
    success,image = vidcap.read()
    print('Dimensiones originales : ',image.shape)
    scale_percent = escala # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    while (count <= length):
        if(count%(fps*saltos) == 0):
            # resize image
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(RutaVideo+"%d.jpg" % count, resized)     # save frame as JPEG file  
        success,image = vidcap.read()
        count += 1
    
    print('Resized Dimensions : ',resized.shape)


    fin = time.time()
    print("TIME : %d [seg]" % round(fin-inicio, 2)) 
    print("----------------------------------------")
    return rutaFrames, VideoName

def isame(rute1, rute2, dbugg = False):
    # COlOR
    # im1 = cv2.imread(rute1)
    # im2 = cv2.imread(rute2)
    #BLANCO Y NEGRO
    im1 = cv2.imread(rute1, 0)
    im2 = cv2.imread(rute2, 0)
    im1F = img_as_float(im1)
    im2F = img_as_float(im2)
    
    # Aplicando metrica SSIM
    # ssimV = ssim(im1F, im2F, multichannel=True, data_range=im2F.max() - im2F.min())
    ssimV = ssim(im1F, im2F, multichannel=False, data_range=im2F.max() - im2F.min())
    dif = np.sum(im1 != im2)

    # Dimensiones imagen
    height = im1.shape[0]
    width = im1.shape[1]
    # channels = im1.shape[2]
    pix_num = height * width * 3

    state = False
    if ( dif/pix_num < 0.001):
        #  Son escencial- la misma
        print(" ----------------- dif %f ----------------- " % float(dif/pix_num))		
        state  = True
    elif(ssimV>0.999):	
        print(" ----------------- ssimV %f ----------------- " % ssimV)		
        # Son escencial- la misma
        state  = True

    if (dbugg):
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(im1)
        f.add_subplot(1,2, 2)
        plt.imshow(im2)
        plt.title("SAME ? :" + str(state) )
        plt.show(block=True)

    return state

# f_ruta = "C:/Users/FrancoPalma/Desktop/PROTOTIPO/video2/F_PitchLifetech_360p/"
# frame = ".jpg"

# Frames = ls(ruta = f_ruta)
# Frames.sort()
# Frames = list(map(addJ ,Frames))
# for i, names in enumerate(Frames):
# 	if(i > 9 and i < 14):
# 		print(i, names)
# 		print(isame(f_ruta+Frames[i-1], f_ruta+names))
# 		# exit(1)

###################################### FUNCIONES EXTRA 2 ######################################
def mar(Oruta, Druta): # Move and Rename files 
    '''
        Oruta : ruta origen, donde estas separas en distintas carpetas
        Druta: ruta destino, donde se mergen todo
        Funcion para mover archivos deesde carpetas pares_<name> -> Data/1, Data/2, Data/3, Data/4
    '''
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    for file in os.listdir(Oruta):	# recorriendo pares
        if (os.path.isdir(Oruta+"/"+file) ): 
            for file2 in os.listdir(Oruta+"/"+file):	# recorriendo 1, 2, 3, 4
                if(file2 == "1"):
                    aux = str(c1)
                elif(file2 == "2"):
                    aux = str(c2)
                elif(file2 == "3"):
                    aux = str(c3)
                elif(file2 == "4"):
                    aux = str(c4)
                if (os.path.isdir(Oruta+"/"+file+"/"+file2) ):
                    for file3 in os.listdir(Oruta+"/"+file+"/"+file2):	# recorriendo frames
                        # print(file3)						
                        # name = "nombre carpeta origen"
                        # Nname = "nombre carpeta destino"						
                        os.rename(Oruta+"/"+file+"/"+file2+"/"+file3, Druta+"/"+file2+"/"+aux+'.jpg')
                        ## RESIZE
                        # width = 1024
                        # height = 288
                        # img=cv2.imread(Druta+"/"+file2+"/"+aux+'.jpg', cv2.IMREAD_GRAYSCALE)
                        # img=cv2.resize(img, (width, height))
                        if(file2 == "1"):
                            c1 = c1 + 1
                            aux = str(c1)
                        elif(file2 == "2"):
                            c2 = c2 + 1
                            aux = str(c2)
                        elif(file2 == "3"):
                            c3 = c3 + 1
                            aux = str(c3)
                        elif(file2 == "4"):
                            c4 = c4 + 1
                            aux = str(c4)
                    # exit(1)
                        # os.rename(Oruta+"/"+file+"/"+file2+"/"+'aux.jpg', Nname)

def mar2(Oruta, Druta): # Move and Rename files 
    '''
        Oruta : ruta origen, donde estas separas en distintas carpetas
        Druta: ruta destino, donde se mergen todo
        Funcion para mover archivos deesde carpetas pares_<name> -> Data/1, Data/2, Data/3, Data/4
    '''
    c1 = 0
    c2 = 0
    for file in os.listdir(Oruta):	# recorriendo pares
        if (os.path.isdir(Oruta+"/"+file) ): 
            for file2 in os.listdir(Oruta+"/"+file):	# recorriendo 1, 2, 3, 4
                if(file2 == "igual"):
                    aux = str(c1)
                elif(file2 == "diferente"):
                    aux = str(c2)
                if (os.path.isdir(Oruta+"/"+file+"/"+file2) ):
                    for file3 in os.listdir(Oruta+"/"+file+"/"+file2):	# recorriendo frames					
                        os.rename(Oruta+"/"+file+"/"+file2+"/"+file3, Druta+"/"+file2+"/"+aux+'.jpg')
                        if(file2 == "igual"):
                            c1 = c1 + 1
                            aux = str(c1)
                        elif(file2 == "diferente"):
                            c2 = c2 + 1
                            aux = str(c2)

###################################### FUNCIONES EXTRA 2 ######################################

def clean(f_ruta, nombre):
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))
    prefijo = "clean" 
    nombre = "./%s_%s" % (prefijo, nombre)

    if (not os.path.isdir(nombre)):
        os.mkdir(nombre)

    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    j = 0 
    for a, frame in enumerate(Frames):
        i = int(frame.split(".")[0]) 
        if(a != 0):
            rute1 = f_ruta+ str(anterior)+'.jpg'
            rute2 = f_ruta+ str(i)+'.jpg'
            if(not isame(rute1, rute2)):
                print("Frame: %d" % j)
                shutil.copy(rute1, nombre)
                j += 1
        anterior = i
    return "OK"

def clean_a(f_ruta, nombre): # retorna array con posiciones de los frames filtrados
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))
    # prefijo = "clean" 
    # nombre = "./%s_%s" % (prefijo, nombre)

    # if (not os.path.isdir(nombre)):
    # 	os.mkdir(nombre)
    pos = []
    full = []
    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    j = 0 
    for a, frame in enumerate(Frames):
        full.append(a)
        i = int(frame.split(".")[0]) 
        if(a != 0):
            rute1 = f_ruta+ str(anterior)+'.jpg'
            rute2 = f_ruta+ str(i)+'.jpg'
            if(not isame(rute1, rute2)):
                print("Frame: %d" % j)
                pos.append(a-1)
                # shutil.copy(rute1, nombre)
                j += 1
        anterior = i
    return pos, full

def getqua(rute1, rute2, me = 1):
    # COlOR
    # im1 = cv2.imread(rute1)
    # im2 = cv2.imread(rute2)
    #BLANCO Y NEGRO
    im1 = cv2.imread(rute1, 0)
    im2 = cv2.imread(rute2, 0)
    im1F = img_as_float(im1)
    im2F = img_as_float(im2)

    height = im1.shape[0]
    width = im1.shape[1]
    pixT = height *  width

    if(me == 1 ):
        # Aplicando metrica SSIM
        # ssimV = ssim(im1F, im2F, multichannel=True, data_range=im2F.max() - im2F.min())
        ssimV = ssim(im1F, im2F, multichannel=False, data_range=im2F.max() - im2F.min())
        return ssimV
    elif(me == 2):
        dif = np.sum(im1 != im2)
        return dif/pixT

def getdata(f_ruta):
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))

    # data = list()
    data = np.array([])

    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    for a, frame in enumerate(Frames):
        i = int(frame.split(".")[0]) 
        if(a != 0):
            rute1 = f_ruta+ str(anterior)+'.jpg'
            rute2 = f_ruta+ str(i)+'.jpg'
            qua =  getqua(rute1, rute2, 1) # SSIM
            # qua =  getqua(rute1, rute2, 2) 
            # print(qua)
            # TEST grafico
            # if (qua > 0.9):
            # 	qua = 1
            data = np.append(data, qua) 

        anterior = i
    return data

def getdata_selec(f_ruta, pos_array):
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))

    # data = list()
    data = np.array([])

    if (len(pos_array) == 1):
        data = np.append(data, 1)
        return data

    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    for a, frame in enumerate(Frames):
        i = int(frame.split(".")[0]) 
        if((a != 0) and (a != pos_array[0]) and ( a in pos_array) ):
            rute1 = f_ruta+ str(anterior)+'.jpg'
            rute2 = f_ruta+ str(i)+'.jpg'
            qua =  getqua(rute1, rute2, 1) # SSIM
            # qua =  getqua(rute1, rute2, 2) 
            # print(qua)
            # TEST grafico
            # if (qua > 0.9):
            # 	qua = 1
            data = np.append(data, qua) 

        anterior = i
    return data

def ploteo(f_ruta, nombre, data = []):

    if (len(data) == 0):
        data = getdata(f_ruta)

    # data = getdata(f_ruta)
    # histograma(data, nombre)
    # print(len(data))
    print("#####################")
    print(nombre)
    # min, minl = localmin(data)
    # print(min)
    print("#####################")
    classic(data, nombre)
    return "OK"

def localmin(data):
    a_min =  np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]
    a_max =  np.r_[True, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], True]
    unique, counts = np.unique(a_min, return_counts=True)
    pos = []
    for index, i in enumerate(a_min):
        if(i):
            pos.append(index)
    # print(counts, len(data))
    # print(pos)
    # exit(1)
    return counts[1], pos

def classic(data, nombre):
    minim = np.amin(data)
    maxim = np.amax(data)
    # print(data)
    # np.log(data)
    # plt.plot(np.exp(data), label='exp.data', color='r')
    plt.plot(data, label='data', color='b')
    # plt.plot(np.log(data), label='log.data', color='g')
    plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
    plt.title("%s (%f,%f)" % (nombre, minim, maxim))
    plt.show()

def histograma(data, nombre):
    plt.title(nombre)
    plt.hist(data, bins=60, alpha=1, edgecolor = 'black',  linewidth=1)
    plt.grid(True)
    plt.show()
    plt.clf()
                
def get_setslides(f_ruta, data = []):
    """
    Funcion que separa las diapositivas en listas de frames dentro de un diccionario
    data = array con las posiciones de los frames que se filtraron ()
    path = ruta desde el archivo 
    Retorna diccionario
    """ 
    # adiv = array con las posiciones donde se divide las diapositivas
    # la data es opcional, ya que se puede sacar directamente del path
    # pero si viene la data entonces esta viene filtrada (suposicion)

    if (len(data) == 0):
        data = getdata(f_ruta)

    N = len(data) + 1

    num_slides, pos_division = localmin(data)
    # sets = dict()
    sets = []
    pos_division.append(N)

    j = 0
    array = []
    for i in range(N):
        if (i <= pos_division[j]):
            array.append(i)
        else:
            # sets[j] = array
            sets.append(array)
            j += 1
            array = []
            array.append(i)
            
    # sets[j] = array
    sets.append(array)
    # print(sets)
    return(sets)

def last_ones(array):
    largo = len(array)
    retorno =  []
    for i in range(largo):
        retorno.append(array[i][-1])
    return retorno

def easy(ruta, detail):
    
    reader = easyocr.Reader(['en'], gpu=False) # this needs to run only once to load the model into memory
    result = reader.readtext(ruta, detail = detail)
    if (detail == 1):
        im = Image.open(ruta)
        # Create figure and axes
        fig_dims = (5, 5)
        fig, ax = plt.subplots(figsize=fig_dims)
        # Display the image
        ax.imshow(im)
        ejex = 0
        ejey = 0
        trans = ""
        ref_pos = []
        ref_text = ""
        c = 0
        for p, t, a in result :
            aux = []
            count = 0
            trans = trans + t + "\n"
            for  pos, text, accu in result :			
                if (c < count): 
                    dis = round(min_dis_sq(p, pos),2)
                    aux.append(dis)


                if ( pos[2][0] > ejex): 
                    ejex = pos[2][0] 
                if ( pos[2][1] > ejey): 
                    ejey = pos[2][1] 
                ancho = pos[1][0] - pos[0][0]
                alto = pos[2][1] - pos[1][1]
                x, y =  pos[0]
                # Create a Rectangle patch
                # rect = patches.Rectangle((x, y), ancho, alto, linewidth=1, edgecolor='r', facecolor='none')
                rect = patches.Polygon(pos, linewidth=1, edgecolor='r', facecolor='none')
                plt.text(x, y,str(count))
                # Add the patch to the Axes
                ax.add_patch(rect)
                count+= 1
            c += 1
            ref_pos.append(aux)
        ax.set_xlim(0, ejex+50)
        ax.set_ylim(0, ejey+50)
        ax.invert_yaxis()
        # fin = time.time()
        # print("TIME : %d [seg]" % round(fin-inicio, 2)) 
        # print(ref_pos)
        # print(len(ref_pos))
        clusters = clustering(ref_pos)
        clustr = str(clusters)
        print(clusters)
        # exit(1)

        print("#############################################")
        # ORDEN DE LOS PUNTOS SEGUN POSISICION HORIZONTAL 
        # clus 
        orden_l = sorted([item for sublist in clusters for item in sublist])
        pos_l = [p[0][0] for (p, t, a) in result]
        zip_list = list(zip(pos_l, orden_l))
        zip_sort = sorted(zip_list, key=lambda x: x[0])
        order_X = [i[1] for i in zip_sort ]
        # print(clusters)
        # print(orden_l)
        # print(pos_l)
        print(zip_list)
        print(zip_sort)
        # print(order_X)
        # exit(1)
        # aux = [k[0] for kinde, k in enumerate(result) if kinde in i] 
        order_Y = []
        for index, i in enumerate(clusters):
            if (len(i)> 1):
                clus = []
                aux = [k[0] for kinde, k in enumerate(result) if kinde in i]  # lista de pos in cluster i
                # print("aux")
                # print(aux) 
                # exit(1)
                lis = [k[0] for k in aux] # lista de pos1 del cluster i
                lis3 =  [k[3] for k in aux] # lista de pos3 del cluster i
                # print(lis)
                # list(num for sublist in array2 for num in sublist)
                list_L = [k[0] for k in lis] # lista de pos1.x
                list_H = [k[1] for k in lis] # lista de pos1.y
                list_h = [k[1] for k in lis3] # lista de pos3.y
                # list_H[2] = 76
                # print(list_H)
                # print(list_h)
                # print(list_L)
                # higher = min(list_H) # valor mas alto 
                # high = list_h[list_H.index(higher)] # valor mas alto 
                # list_H[list_H.index(higher)] = float('inf')
                # print(list_H)
                # exit(1)

                lefter = min(list_L) # valor mas izquierdo
                # print(higher, high)
                # print(lefter)
                # rango =  (high- higher)/2 # valor medio de la altura # ------------------------- ME FALTA TOMAR EL PUNTO 1 Y EL 3 O 4 PARA MEDIR LA ALTURA  (QUIZAS TENGA PROBLEMA CON LOS RECTANGULOS DIAGONALES)
                # print(rango)
                # print(float('inf') )
                # print("while")
                
                order = []
                while(len(i) > len([item for sublist in clus for item in sublist])):
                    # order.append(i[list_L.index(lefter)])
                    # list_L[list_L.index(lefter)] = float('inf')
                    # lefter = min(list_L) 
                    higher = min(list_H) # valor mas alto 
                    pos_H = list_H.index(higher)
                    high = list_h[pos_H] # valor mas alto 
                    list_H[pos_H] = float('inf')
                    rango =  (high- higher)/4
                    levels = []
                    # print("uno")
                    levels.append(i[pos_H])
                    for jndex, j in enumerate(i): # set(range(tot)) - set([i])
                        # print(levels)
                        if(higher+rango > list_H[jndex] ):
                            # print("dos")
                            levels.append(i[jndex])
                            list_H[jndex] = float('inf')
                        # levels.append()
                        # p, t, a = result[j]
                        # p = list(map((lambda x: [round(x[0], 2), round(x[1], 2)] ), p ))
                        # print(j)

                    clus.append(levels)
            else: # CASO EN QUE len(i) == 1
                clus = i
                # print("tres")
                    # exit(1)
            # print(clus)
            order_Y.append(clus)
        print(order_Y)
        print("### for ###")
        print("orderX",order_X)
        order = order_Y
        for index, i in enumerate(order_Y):
            if(len(i) > 1):
                for jndex, j in enumerate(i):
                    if(len(j) > 1):
                        x_ord = [x for x in order_X if x in j]
                        order[index][jndex] = x_ord
                        print(j)
                        print(x_ord)
        print("### for ###")
        print(order)
        print("#############################################")
        
        filename = "order.json"
        # order_trans = str(order).replace(i,result[i])
        order_trans = str(order) 
        # TODO: RECORRER order, Y SOBRE ESE REEMPLAZAR CON t 
        # TODO: ARREGLAR FOTO WSP
        # TODO: REVISAR QUE TAN UTIL SERIA EL RTF
        # TODO: AVANZAR DOCUMENTO, ORDENAR CODIGO, 
        for index, (p, t, a) in enumerate(result):
            p = list(map((lambda x: [round(x[0], 2), round(x[1], 2)] ), p ))
            order_trans = order_trans.replace(str(index), t)
            # print(p, t)
        write_json(order_trans, filename)
        plt.show()
        exit(1)
        # print(clusters)
        # print(clustr)
        # exit(1)
        return trans
    else:
        return (" ").join(result)

def get_indexs(largo, index):
    """ Funcion obtiene indices a  partir de indice de array2 flaten
    """
    indx = 0
    indy = 0
    for i in range(largo):
        #LOGICA para obtener los indices
        print()
    return indx, indy

def deep_index(lst, w):
    l = list((i, sub.index(w)) for (i, sub) in enumerate(lst) if w in sub)
    return l[0]

def clustering(array2):
    """ Funcion forma grupos segun distancias entregadas
    args: 
    array2 (array(array(int))): lista de listas con distancias entre bloques de texto en un frame
    returns:
    ret_array2 (array(array(int))): lista de grupos creados a partir de las distancias
    """
    tot = len(array2)
    aux = [[None]]*tot
    ret_array2 = [[None]]*tot
    flatten = list(num for sublist in array2 for num in sublist)
    maxim = max(flatten) 
    tot_flat = len(flatten) 
    average = sum(flatten)/tot_flat  # TODO: PROBAR MEDIANA en vez de media
    fla_sort = sorted(flatten)
    media = fla_sort[floor(tot_flat/2)]
    metrica = media# average o media
    while ( len(list(num for sublist in ret_array2 for num in sublist)) < tot*2):  ## TODO: limita el numero de links validos, quizas usar otro limite
        minim = min(flatten)
        if (minim > metrica):
            #Agregar solos los que no alcanzaron en ret_array2 
            print("solitos")
            print(flatten[flatten != maxim+1])
            break
        ind = flatten.index(minim) 
        indx, indy = deep_index(array2, minim)        

        flatten[ind] = maxim + 1
        array2[indx][indy] = maxim + 1
        ret_array2[indx] = ret_array2[indx] + [indy+indx+1]


    for j in range(tot):
        if(len(ret_array2[j]) > 1):
            ret_array2[j][0] = j
        elif(len(ret_array2[j]) == 1):
            ret_array2[j] = [j]

    for i in range(tot):
        if(len(ret_array2[i]) == 0):
            continue
        for j in  set(range(tot)) - set([i]): # range(tot):
            if(len([i for i in ret_array2[i] if i in ret_array2[j]]) > 0):
                ret_array2[i] = list(set(ret_array2[i] + ret_array2[j]))
                ret_array2[j] = []


    ret_array2 = [i for i in ret_array2 if len(i)>0]
    # print(f"ret_array2 : {ret_array2} size : {len(ret_array2)}")
    return(ret_array2)
            
def write_json(data, filename= "adw.json"):
    with  open(filename, "w") as f:
        json.dump(data, f, indent=4)

def tese(ruta, debug = False):
    inicio = time.time()
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = cv2.imread(ruta, 0)
    conf = f'--psm 6'
    data = pytesseract.image_to_string(image, lang='eng', config=conf)
    if (debug == True):
        # plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
        # plt.imshow(opening, cmap = 'gray', interpolation = 'bicubic')
        plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        fin = time.time()
        print("TIME : %d [seg]" % round(fin-inicio, 2)) 
    return data

def get_transcription(f_ruta, data = [], ocr = 1): # 1 = easyOCR y 2 = teseract
    Frames = ls(ruta = f_ruta)
    Frames.sort()
    Frames = list(map(addJ ,Frames))
    inicio = time.time()

    transcription = ""
    # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
    f_ruta = f_ruta+"/"
    for a, frame in enumerate(Frames):
        # if(a != 5): # ELIMINAR
        #     # print(data, type(data))
        #     data 
        #     continue
        i = int(frame.split(".")[0]) 
        if (len(data) != 0 and a in data):
            # if(a != 0):
            rute = f_ruta+ str(i)+'.jpg'
            if (ocr == 1):
                transcription = transcription + easy(rute, 1) + "\n\n"
            elif (ocr == 2):
                transcription = transcription + tese(rute, False) + "\n\n"
    
    fin = time.time()
    print("TIME : %d [seg]" % round(fin-inicio, 2)) 
    return transcription

def dist_2p(pos1, pos2):
    return sqrt( (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 )

def min_dis_sq(pos1, pos2): # arrays con coordenadas de ambos cuadrados
    a1,a2,a3,a4 = pos1
    b1,b2,b3,b4 = pos2
    #  usar puntos 1 y 3, ya que son de lados iguales
    dist = 0
    if (a3[0] < b1[0]): # B esta completamente a la derecha de A -> A<<B 
        # print("CASO 1")
        if (a1[1] > b3[1]):            
            # print(f"A : {dist_2p(a2, b4)}")
            return(dist_2p(a2, b4))
        elif (b1[1] > a3[1]):
            # print(f"G : {dist_2p(a3, b1)}")
            return(dist_2p(a3, b1))
        # elif ( b1[1] <= a1[1] <= b3[1] or  b1[1] <= a3[1] <= b3[1]):
        else:
            # print(f"BCDEF : {b1[0] - a3[0]}")
            return(b1[0] - a3[0])
        # else:
        #     print("FALLO 1")

    elif (b3[0] < a1[0]):
        # print("CASO 7")
        if (a1[1] > b3[1]):
            return(dist_2p(a1, b3))
            print(f"A : {dist_2p(a1, b3)}") 
        elif (b1[1] > a3[1]):
            return(dist_2p(a4, b2))
            print(f"G : {dist_2p(a4, b2)}")
        # elif ( b1[1] <= a1[1] <= b3[1] or  b1[1] <= a3[1] <= b3[1]):
        else:
            return(a1[0] - b3[0])
            print(f"BCDEF : {a1[0] - b3[0]}")
        # else:
        #     print("FALLO 2")


    elif ( b1[0] <= a1[0] <= b3[0] or  b1[0] <= a3[0] <= b3[0] or a1[0] <= b1[0] <= a3[0] or  a1[0] <= b3[0] <= a3[0]):
        # print("2, 3, 4, 5, 6")
        if (a1[1] > b3[1]):
            return(a1[1] - b3[1])
            print(f"A : {a1[1] - b3[1]}")
        elif (b1[1] > a3[1]):
            return(b1[1] - a3[1])
            print(f"G : {b1[1] - a3[1]}")
        else:
            return(dist)
            print(f"resto: {dist}")

    else:
        print("FALLO 3")
################################ MAIN ################################

# mar2('C:/Users/FrancoPalma/Downloads/IMAGENES2-20220408T005748Z-001/Data(issame)', 'C:/Users/FrancoPalma/Downloads/IMAGENES2-20220408T005748Z-001/Data')
# exit(1)

# mar('C:/Users/FrancoPalma/Downloads/IMAGENES2-20220408T005748Z-001/IMAGENES2', 'C:/Users/FrancoPalma/Downloads/IMAGENES2-20220408T005748Z-001/Data')


# RC10 = "./video2/y2mate.com - Almacén Digital_360p.mp4"  # frames vienen con problemas
# RC = "./video2/Data Health.mp4" # tamaño de los frames es muy grande
RC2 = "./video2/Closet Cleanup_360p.mp4.webm"    
RC3 = "./video2/Estrategia Digital MBA UC examen_360p.mp4"
RC4 = "./video2/MBAUC  Q22021  Estrategia Digital  Grow  Invest_360p.mp4"
RC5 = "./video2/Pitch Lifetech_360p.mp4"
RC6 = "./video2/PLATAFOMRA DE SEGUROSEST DIGITAL_360p.mp4"
RC7 = "./video2/Presentacion   TRADE NOW_360p.mp4.webm"
RC8 = "./video2/Un alivio a un click de distancia_360p.mp4"
RC9 = "./video2/VESKI_360p.mp4"

# parametrizar de tal forma de pasarse en el numero de diapos en vez de que falten ( menor perdida )
################# OBTENER FRAMES #################
saltos = 2	 	# 0, 1, 2, 3 ...
escala = 100		# tamano final de la imagen (en porcentaje)
fname = "Default" 	# 
lista_rutas = [RC2, RC3, RC4, RC5, RC6, RC7, RC8, RC9]
# for i, ruta in enumerate(lista_rutas):
    # print(i, ruta)
    # rutaFrames, nombreVideo = getFrames(ruta, saltos, escala)

# rutaFrames, nombreVideo = "./video2/F_ClosetCleanup_360pwebm", "ClosetCleanup_360pwebm"
# rutaFrames, nombreVideo = "./video2/F_EstrategiaDigitalMBAUCexamen_360p", "EstrategiaDigitalMBAUCexamen_360p"
# rutaFrames, nombreVideo = "./video2/F_MBAUCQ22021EstrategiaDigitalGrowInvest_360p", "MBAUCQ22021EstrategiaDigitalGrowInvest_360p"
# rutaFrames, nombreVideo = "./video2/F_PitchLifetech_360p", "PitchLifetech_360p"
# rutaFrames, nombreVideo = "./video2/F_PLATAFOMRADESEGUROSESTDIGITAL_360p", "PLATAFOMRADESEGUROSESTDIGITAL_360p"
# rutaFrames, nombreVideo = "./video2/F_PresentacionTRADENOW_360pwebm", "PresentacionTRADENOW_360pwebm"
# rutaFrames, nombreVideo = "./video2/F_Unalivioaunclickdedistancia_360p", "Unalivioaunclickdedistancia_360p"
# rutaFrames, nombreVideo = "./video2/F_VESKI_360p", "VESKI_360p"

RC2, NV2 = "../CLEAN/clean_ClosetCleanup_360pwebm" 						, "clean_ClosetCleanup_360pwebm"
RC3, NV3 = "../CLEAN/clean_EstrategiaDigitalMBAUCexamen_360p"			, "clean_EstrategiaDigitalMBAUCexamen_360p"
RC4, NV4 = "../CLEAN/clean_MBAUCQ22021EstrategiaDigitalGrowInvest_360p"	, "clean_MBAUCQ22021EstrategiaDigitalGrowInvest_360p"
RC5, NV5 = "../CLEAN/clean_PitchLifetech_360p"							, "clean_PitchLifetech_360p"
RC6, NV6 = "../CLEAN/clean_PLATAFOMRADESEGUROSESTDIGITAL_360p"			, "clean_PLATAFOMRADESEGUROSESTDIGITAL_360p"
RC7, NV7 = "../CLEAN/clean_PresentacionTRADENOW_360pwebm"				, "clean_PresentacionTRADENOW_360pwebm"
RC8, NV8 = "../CLEAN/clean_Unalivioaunclickdedistancia_360p"				, "clean_Unalivioaunclickdedistancia_360p"
RC9, NV9 = "../CLEAN/clean_VESKI_360p"									, "clean_VESKI_360p"

lista_rutas_clean = [RC2, RC3, RC4, RC5, RC6, RC7, RC8, RC9]
lista_nombreVideo = [NV2, NV3, NV4, NV5, NV6, NV7, NV8, NV9]

# for i, ruta in enumerate(lista_rutas_clean):
    # print(ruta, lista_nombreVideo[i])
    # ploteo(ruta, lista_nombreVideo[i] )  # grafica 
    # clean( ruta, lista_nombreVideo[i] )



rutaFrames, nombreVideo = "../CLEAN/clean_ClosetCleanup_360pwebm", "clean_ClosetCleanup_360pwebm"
# rutaFrames, nombreVideo = "./CLEAN/clean_EstrategiaDigitalMBAUCexamen_360p", "clean_EstrategiaDigitalMBAUCexamen_360p"
# rutaFrames, nombreVideo = "./CLEAN/clean_MBAUCQ22021EstrategiaDigitalGrowInvest_360p", "clean_MBAUCQ22021EstrategiaDigitalGrowInvest_360p"

sets = get_setslides(rutaFrames)
# print(sets)
# print()
print(" #################### sets #################### ")
print(sets)
seleccionados =  last_ones(sets)
print(" #################### seleccionados #################### ")
print(seleccionados)
# print(seleccionados)
get_transcription(rutaFrames, data = seleccionados, ocr = 1)
exit(1)
for i in range(len(sets)):
    data = getdata_selec(rutaFrames, sets[i])
    print(sets[i])
    print(data)
    # ploteo(rutaFrames, nombreVideo, data)

pos , full = clean_a(rutaFrames, nombreVideo)
print(pos)
print(full)
exit(1)