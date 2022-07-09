import os
import time
import cv2
from pytube import YouTube
import validators
from pytube import YouTube
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from matplotlib import pyplot as plt
import re

def addJ (name):
    # return "test/"+str(name)+".jpg"
    return str(name)+".jpg"

def ls(ruta = Path.cwd()):
    """ Funcion que obtiene Lista de nombres de Frames casteado a entero
    -------------------------------------------------------
    Input:
        ruta (str): ruta de carpeta donde se encuentran los frames
    Output:
        (array(int)) Lista de nombres de Frames casteado a entero
    """
    return [int(arch.name.split(".")[0]) for arch in Path(ruta).iterdir() if (arch.is_file() and re.search(r'\.jpg$', arch.name))]

def download_video(url): 
    """ Descarga un video de youtube 
    -------------------------------------------------------
    Input:
        url (str): link del video de youtube
    Output:
        No aplica
    """
    ''' 
    CAMBIOS QUE REQUIERE EN CIPHER.PY (DOCUMENTOS DE LA LIBRERIA)
    lineas 272 y 273
    r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
    r'\([a-z]\s*=\s*([a-zA-Z0-9$]{2,3})(\[\d+\])?\([a-z]\)'
    cambiar linea  288
    nfunc=re.escape(function_match.group(1))),
    '''
    # for stream in video.streams:
    #     print(stream)
    try:
        video = YouTube(url)
        title = video.title
        video = video.streams.get_highest_resolution()
        video.download()
        return True, title
    except:
        return False, ""

def getqua(rute1, rute2, me = 1): 
    """ Funcion que compara dos frames con la metrica que indica el parametro "me"
    -------------------------------------------------------
    Input:
        f_ruta (str): ruta frames
        nombre (str): nombre de la extension de la carpeta de bloques
        me (int): metrica a usar para comparar los frames
    Output:
        Valor (float): valor de evaluacion obtenido con metrica elegida
    """
    # COlOR
    # im1 = cv2.imread(rute1)
    # im2 = cv2.imread(rute2)
    #BLANCO Y NEGRO
    if(isinstance(rute1, str)):
        im1 = cv2.imread(rute1, 0)
        im2 = cv2.imread(rute2, 0)
    else:
        im1 = rute1
        im2 = rute2
    im1F = img_as_float(im1)
    im2F = img_as_float(im2)

    # plt.subplot(1, 2, 1)
    # plt.imshow(im1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(im2)
    # plt.show()

    height = im1.shape[0]
    width = im1.shape[1]
    pixT = height *  width

    if(me == 1 ):
        # Aplicando metrica SSIM
        try:
            ssimV = ssim(im1F, im2F, multichannel=True, data_range=im2F.max() - im2F.min())
        except:
            ssimV = ssim(im1F, im2F, multichannel=False, data_range=im2F.max() - im2F.min())
        return ssimV
    elif(me == 2):
        dif = np.sum(im1 != im2)
        return dif/pixT

def getdata(f_ruta): 
    """ Funcion que usando getqua() en frames ordenados entrega un array con los valores evaluados de frames contiguos
    -------------------------------------------------------
    Input:
        f_ruta (str): ruta frames o (list): array de imagenes cv2
    Output:
        data (list): array ordenado con numeros enteros obtenidos evaluando frames contiguos
    """
    # data = list()
    data = np.array([])

    if(isinstance(f_ruta, list)):
        for index, frame in enumerate(f_ruta):
            if(index != 0):
                rute2 = frame
                qua =  getqua(rute1, rute2, 1) # SSIM
                data = np.append(data, qua) 
                rute1 = rute2
            else:
                rute1 = frame
    else:
        Frames = ls(ruta = f_ruta)
        Frames.sort()
        Frames = list(map(addJ ,Frames))

        # iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
        f_ruta = f_ruta+"/"
        for index, frame in enumerate(Frames):
            i = int(frame.split(".")[0]) 
            if(index != 0):
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

class Video:
    def __init__(self, path, scale, saltos, local = False): # scale:percent of original size
        """ Clase para manejar el video, frames y transcripcion 
        path (str): link del video o a la ruta local del archivo mp4
        scale (int): numero que indica de que escala del tamaño real de los frames se desean extraer [0,100]
        saltos (int): numero de saltos periodicos entre lecturas de frames
        local (boolean): indicador para usar la data de frames de forma persistente (archivos) o en ejecucion (objetos y listas)
        """
        link = True
        # ------------ Video de Youtube ------------
        if (validators.url(path)):
            status, real_VideoName = download_video(path)
            real_VideoName = real_VideoName.replace("|", "")
            if(not status):
                raise Exception("El link entregado no es un video")
            RutaFolder = os.path.dirname(os.path.abspath(__file__))+"\\"
            self.path = RutaFolder+real_VideoName+".mp4"
            self.video_name = real_VideoName
        # ------------------------------------------
        # ------------ Video desde directorio ------------
        else:
            link = False
            real_VideoName = path.split("/")[-1] 
            RutaFolder = path.replace(real_VideoName, '')
            self.path = path
            self.video_name = real_VideoName.replace(".mp4", "") #.replace("y2mate.com", "").replace(" ", "").replace(".", "").replace("-", "")
        # ------------------------------------------------
            
        # ------------ Se crea carpeta y se captura el video ------------
        self.frames_path = RutaFolder+"F_"+self.video_name+"\\"
        if (not os.path.isdir(self.frames_path)):
            os.mkdir(self.frames_path)
        vidcap = cv2.VideoCapture(RutaFolder+self.video_name+".mp4")
        self.video_cap = vidcap
        # ---------------------------------------------------------------
        
        # ------------ Se elimina el video en caso de local y link ------------
        if(local and link): 
            string = """
            Se borra la carpeta con los frames -> solo se puede cuando se deje de usar el vidcap
            se mantiene una lista de imagenes = frames
            se mantiene una vidcap = video
            """ 
            # os.remove(self.path)
            print(string)
        # ---------------------------------------------------------------------

        self.num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps    = int(vidcap.get(cv2.CAP_PROP_FPS))
        count = 0
        # ------------ Se lee un frame y se obtiene las dimensiones ------------
        success,image = vidcap.read()
        print(f"success: {success}")
        print(f'Dimensiones originales : {image.shape}')
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)
        self.dim = dim
        # ----------------------------------------------------------------------

        # ------------ Se guardan los frames o se crea lista de frames ------------
        frames = []
        while (count <= self.num_frames):
            if(count%(self.fps*saltos) == 0):
                # resize image
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                if(local):
                    frames.append(resized)
                else:
                    cv2.imwrite(self.frames_path+"%d.jpg" % count, resized)     # save frame as JPEG file  
            success,image = vidcap.read()
            count += 1

        self.frames = frames
        # -------------------------------------------------------------------------

    # --------------- GETTERS ---------------
    def get_number_frames(self):  # numero de frames
        return self.num_frames
    def get_fps(self):            # fotogramas por segundo del video
        return self.fps
    def get_path(self):           # ruta del video
        return self.path
    def get_video_name(self):     # nombre del video
        return self.video_name
    def get_frames_path(self):    # ruta de los frames
        return self.frames_path
    def get_video_cap(self):      # captura del video 
        return self.video_cap
    def get_frames(self):
        return self.frames
    # --------------- SETTERS ---------------
    def set_frames_path(self, frames_path):
        self.frames_path = frames_path

    def set_slides(self, posiciones = None):
        """ divide y obtiene los frames que contienen la mayor parte de la informacion de cada slide
        -------------------------------------------------------
        Input:
            posiciones (array): lista con posiciones de los frames elegios para conformar el conjunto final de diapositivas
        Output:
            No aplica
        """
        if(posiciones != None):
            self.frames = [i for index, i in enumerate(self.frames) if index in posiciones]
        else:
            print("obtener diapositivas")





RC8 = "./video2/y2mate.com - Un alivio a un click de distancia_360p.mp4"
# video1 = Video(RC8, 100, 1)
# getFrames(ruta, saltos, escala = 100 ,fname = "Default"):

# print(video1.get_number_frames())
# print(video1.get_fps())
# print(video1.get_path())
# print(video1.get_video_name())
# print(video1.get_frames_path())
# print(video1.get_video_cap())
# saltos acerlo proporcional al numero de frames

RC1 = "./video2/Data Health.mp4"
RC2 = "./video2/y2mate.com - Closet Cleanup_360p.mp4.webm"
RC3 = "./video2/y2mate.com - Estrategia Digital MBA UC examen_360p.mp4"
RC4 = "./video2/y2mate.com - MBAUC  Q22021  Estrategia Digital  Grow  Invest_360p.mp4"
RC5 = "./video2/y2mate.com - Pitch Lifetech_360p.mp4"
RC6 = "./video2/y2mate.com - PLATAFOMRA DE SEGUROSEST DIGITAL_360p.mp4"
RC7 = "./video2/y2mate.com - Presentacion   TRADE NOW_360p.mp4.webm"
RC8 = "./video2/y2mate.com - Un alivio a un click de distancia_360p.mp4"
RC9 = "./video2/y2mate.com - VESKI_360p.mp4"
RC10 = "./video2/y2mate.com - Almacén Digital_360p.mp4"   # PROBLEMAS ?

string = "http://google.com"

string = "https://youtu.be/5GJWxDKyk3A" 
string = "https://youtu.be/1KmlriQpkXs"  # 1 minuto y medio
directorio = "C:/Users/FrancoPalma/Desktop/PROTOTIPO/T/Billie Eilish - Happier Than Ever (Official Music Video).mp4"

video1 = Video(string, 100, 1)

# print(getdata(video1.frames)) # caso local 
print(getdata(video1.frames_path)) # caso NO local
exit(1)

print( "getqua")
print( getqua(video1.frames[0], video1.frames[1], me = 1) )
print( getqua(video1.frames[1], video1.frames[2], me = 1) )

fold0 = "C:/Users/FrancoPalma/Desktop/PROTOTIPO/T/F_Billie Eilish - Happier Than Ever (Official Music Video)/0.jpg"
fold1 = "C:/Users/FrancoPalma/Desktop/PROTOTIPO/T/F_Billie Eilish - Happier Than Ever (Official Music Video)/23.jpg"
fold2 = "C:/Users/FrancoPalma/Desktop/PROTOTIPO/T/F_Billie Eilish - Happier Than Ever (Official Music Video)/46.jpg"

print( getqua(fold0, fold1, me = 1) )
print( getqua(fold1, fold2, me = 1) )
# pos = [2,5,6]
# video1.set_slides(pos)

# video1 = Video(RC1, 100, 1)
# video2 = Video(RC2, 100, 1)
# video3 = Video(RC3, 100, 1)
# video4 = Video(RC4, 100, 1)
# video5 = Video(RC5, 100, 1)
# video6 = Video(RC6, 100, 1)
# video7 = Video(RC7, 100, 1)
# video8 = Video(RC8, 100, 1)
# video9 = Video(RC9, 100, 1)
# video10 = Video(RC10, 100, 1)

# print(video1.get_fps())
# print(video2.get_fps())
# print(video3.get_fps())
# print(video4.get_fps())
# print(video5.get_fps())
# print(video6.get_fps())
# print(video7.get_fps())
# print(video8.get_fps())
# print(video9.get_fps())
# print(video10.get_fps())





