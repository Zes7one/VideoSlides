import os
import time
import cv2
from pytube import YouTube
import validators
from pytube import YouTube

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
        video = video.streams.get_highest_resolution()
        video.download()
        return True
    except:
        return False

class Video:
    def __init__(self, path, scale, saltos): # scale:percent of original size
        if (validators.url(path)):
            status = download_video(path)
            if(not status):
                raise Exception("El link entregado no es un video")
            path = os.path.dirname(os.path.abspath(__file__))
            exit(1)
            # path = 

        real_VideoName = path.split("/")[-1]
        RutaVideo = path.replace(real_VideoName, '')



        self.path = path
        self.video_name = real_VideoName.replace("y2mate.com", "").replace(".mp4", "").replace(" ", "").replace(".", "").replace("-", "")

        self.frames_path = RutaVideo+"F_"+self.video_name
        if (not os.path.isdir(self.frames_path)):
            os.mkdir(self.frames_path)
        
        vidcap = cv2.VideoCapture(RutaVideo+real_VideoName)
        self.video_cap = vidcap
        self.num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps    = int(vidcap.get(cv2.CAP_PROP_FPS))

        count = 0
        success,image = vidcap.read()
        # print('Dimensiones originales : ',image.shape)
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)

        while (count <= self.num_frames):
            if(count%(self.fps*saltos) == 0):
                # resize image
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(self.frames_path+"%d.jpg" % count, resized)     # save frame as JPEG file  
            success,image = vidcap.read()
            count += 1

    # --------------- GETTERS ---------------
    def get_number_frames(self):
        return self.num_frames
    def get_fps(self):
        return self.fps
    def get_path(self):
        return self.path
    def get_video_name(self):
        return self.video_name
    def get_frames_path(self):
        return self.frames_path
    def get_video_cap(self):
        return self.video_cap
    # --------------- SETTERS ---------------
    def set_frames_path(self, frames_path):
        self.frames_path = frames_path

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
string = "https://youtu.be/47OC5rFeXGs"
video1 = Video(string, 100, 1)

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





