import validators
import warnings
import time
import os
import cv2
import functions as fc

class Video:
    def __init__(self, path, scale = 100 , saltos = 1, local = True): # scale:percent of original size
        """ Clase para manejar el video, frames y transcripcion 
        path (str): link del video o a la ruta local del archivo mp4
        scale (int): numero que indica de que escala del tamaño real de los frames se desean extraer [0,100]
        saltos (int): numero de saltos periodicos entre lecturas de frames
        local (boolean): indicador para usar la data de frames de forma persistente (archivos) o en ejecucion (objetos y listas)
        """
        link = True
        self.local = local
        # ------------ Video de Youtube ------------
        if (validators.url(path)):
            status, real_VideoName = fc.download_video(path)
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
        self.frames_path = ""
        if(not local):
            self.frames_path = RutaFolder+"F_"+self.video_name+"\\"
            if (not os.path.isdir(self.frames_path)):
                os.mkdir(self.frames_path)
        vidcap = cv2.VideoCapture(RutaFolder+self.video_name+".mp4")
        self.video_cap = vidcap
        # ---------------------------------------------------------------
        
        # ------------ Se elimina el video en caso de local y link ------------
        if(local and link): 
            # TODO Se borra la carpeta con los frames -> solo se puede cuando se deje de usar el vidcap
            string = """
            se mantiene una lista de imagenes = frames
            se mantiene una vidcap = video
            """ 
            # os.remove(self.path)
            warnings.warn(string)
        # ---------------------------------------------------------------------

        self.num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps    = int(vidcap.get(cv2.CAP_PROP_FPS))
        # ------------ Se lee un frame y se obtiene las dimensiones ------------
        success,image = vidcap.read()	
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # TODO revisar cambios necesarios para implementar solo escala de grises ( o dar la opcion de elegir)
        if(not success):
            raise Exception("Problemas en la captura del video: video corrompido o formato incorrecto")
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)
        self.dim = dim
        # ----------------------------------------------------------------------

        # ------------ Se guardan los frames o se crea lista de frames ------------
        count = 0
        frames = []
        while (count < self.num_frames-1):
            if(count%(self.fps*saltos) == 0):
                # resize image
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                if(local):
                    frames.append(resized)
                else:
                    cv2.imwrite(self.frames_path+"%d.jpg" % count, resized)     # save frame as JPEG file  
            success,image = vidcap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # TODO revisar cambios necesarios para implementar solo escala de grises ( o dar la opcion de elegir)
            count += 1

        self.frames = frames
        # -------------------------------------------------------------------------

        self.data = []
        self.slides = []

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

    def set_data(self): # se setea la data segun este usandose de forma local o no
        if(self.local):
            self.data = fc.getdata(self.frames) # caso local 
        else:
            self.data = fc.getdata(self.frames_path) # caso NO local

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
            if (len(self.data) == 0):
                self.set_data()
                msg = "No se tiene data, se ejecuta automaticamente el metodo set_data() para setearla en el atributo data"
                warnings.warn(f"Warning........... {msg}")

            N = len(self.data) + 1
            num_slides, pos_division = fc.localmin(self.data)
            sets = []
            pos_division.append(N)

            j = 0
            array = []
            for i in range(N):
                if (i <= pos_division[j]):
                    array.append(i)
                else:
                    sets.append(array)
                    j += 1
                    array = []
                    array.append(i)
                    
            sets.append(array)
            self.slides = fc.last_ones(sets) # Se seleccionan los ultimos frames de cada conjunto 

    def set_transcription(self):
        if (len(self.slides) == 0):
            self.set_slides()
            msg = "No se tienen las slides, se ejecuta automaticamente el metodo set_slides() para setearla en el atributo slides"
            warnings.warn(f"Warning........... {msg}")

        if(self.local):
            self.transcription = fc.get_transcription(self.frames, self.slides) # caso local 
        else:
            self.transcription = fc.get_transcription(self.frames_path, self.slides) # caso NO local

    def clean_frames(self):
        if(self.local):
            self.frames = fc.clean(self.frames)
        else:
            fc.clean(self.frames_path)


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

inicio = time.time()
video1 = Video(string, 100, 1, False)
video1.clean_frames()
video1.set_data()
video1.set_slides() 
video1.set_transcription()


print(video1.data)
print(video1.slides)
print(video1.transcription)
# ploteo(video1.video_name , video1.data) # grafica
fin = time.time()
print("TIME : %d [seg]" % round(fin-inicio, 2)) 
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





