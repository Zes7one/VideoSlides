import os
import warnings

import cv2
import validators
from pathlib import Path
import functions as fc


class Video:
    def __init__(self, path, scale = 100 , saltos = 1, rgb = False, runtime = True, lematiz = False, gpu_use = False, pix_lim = 0.001, ssimv_lim = 0.999): # scale:percent of original size
        """ Clase para manejar el video, frames y transcripcion 
        path (str): link del video o a la ruta local del archivo mp4
        scale (int): numero que indica de que escala del tamaño real de los frames se desean extraer [0,100]
        saltos (int): numero de saltos periodicos entre lecturas de frames
        runtime (boolean): False -> para usar la data de frames de forma persistente (archivos) o True -> en ejecucion (objetos y listas) 
        """
        link = True
        self.rgb = rgb
        self.runtime = runtime
        self.lematiz = lematiz
        self.gpu_use = gpu_use
        self.pix_lim, self.ssimv_lim = pix_lim, ssimv_lim
        # ------------ Video de Youtube ------------
        extension = "mp4"
        if (validators.url(path)):
            status, real_VideoName = fc.download_video(path)
            real_VideoName = real_VideoName.replace("|", "")
            if(not status):
                print(status)
                raise Exception("El link entregado no es un video")
            # RutaFolder = os.path.dirname(os.path.abspath(__file__))+"\\"
            RutaFolder = os.path.abspath(os.getcwd())+"\\"
            self.path = RutaFolder+real_VideoName+f".{extension}"
            self.video_name = real_VideoName
        # ------------------------------------------
        # ------------ Video desde directorio ------------
        else:
            link = False
            real_VideoName = path.split("/")[-1] 
            extension = real_VideoName.split(".")[-1]
            RutaFolder = path.replace(real_VideoName, '')
            self.path = path
            self.video_name = real_VideoName.replace(f".{extension}", "") #.replace("y2mate.com", "").replace(" ", "").replace(".", "").replace("-", "")
        # ------------------------------------------------
            
        # ------------ Se crea carpeta y se captura el video ------------
        self.frames_path = ""
        if(not runtime):
            self.frames_path = RutaFolder+"F_"+self.video_name+"/"
            print('Path(self.frames_path)')
            print(self.frames_path)
            print(Path(self.frames_path))
            if (not os.path.isdir(self.frames_path)):
                os.mkdir(Path(self.frames_path) ) # data_folder = Path("source_data/text_files/")
                # os.mkdir(self.frames_path) # data_folder = Path("source_data/text_files/")
        vidcap = cv2.VideoCapture(RutaFolder+self.video_name+f".{extension}")
        self.video_cap = vidcap
        # ---------------------------------------------------------------
        
        # ------------ Se elimina el video en caso de runtime y link ------------
        # if(runtime and link): 
        #     # TODO Se borra la carpeta con los frames -> solo se puede cuando se deje de usar el vidcap
        #     string = """
        #     Manejo de data en Runtime
        #     se mantiene una lista de imagenes = frames
        #     se mantiene una vidcap = video
        #     """ 
        #     # os.remove(self.path)
        #     warnings.warn(string)
        # ---------------------------------------------------------------------

        self.num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps    = int(vidcap.get(cv2.CAP_PROP_FPS))
        # ------------ Se lee un frame y se obtiene las dimensiones ------------
        success,image = vidcap.read()	
        if(not success):
            raise Exception("Problemas en la captura del video: video corrupto o formato incorrecto")
        if(not rgb):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        dim = (width, height)
        self.dim = dim
        # ----------------------------------------------------------------------

        # ------------ Se guardan los frames o se crea lista de frames ------------
        count = 0
        if(runtime):
            frames = []
        else:
            frames = self.frames_path
        
        while (count < self.num_frames-1):
            if(count%(self.fps*saltos) == 0):
                # resize image
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                if(runtime):
                    frames.append(resized)
                else:
                    # cv2.imwrite(Path(self.frames_path+f"{count}.jpg"), resized)     # save frame as JPEG file  
                    cv2.imwrite(self.frames_path+f"{count}.jpg", resized)     # save frame as JPEG file  
            success,image = vidcap.read()
            if(not rgb):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
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

    def set_data(self, me): # se setea la data segun este usandose de forma runtime o no
        """ Funcion que usando getqua() sobre frames ordenados entrega un array con los valores evaluados de frames contiguos
        -------------------------------------------------------
        Input:
            No aplica
        Output:
            No aplica
        """
        self.data = fc.getdata(self.frames, me, self.rgb) # ambos casos cubiertos (runtime TRUE/FALSE)

    def set_slides(self, me = 1, posiciones = None, coef = 3):
        """ divide y obtiene los frames que contienen la mayor parte de la informacion de cada slide
        -------------------------------------------------------
        Input:
            posiciones (array): lista con posiciones de los frames elegios para conformar el conjunto final de diapositivas (opcional)
        Output:
            No aplica
        """
        if(posiciones != None):
            self.frames = [i for index, i in enumerate(self.frames) if index in posiciones]
        else:
            if (len(self.data) == 0):
                self.set_data(me)
                msg = "No se tiene data, se ejecuta automaticamente el metodo set_data() para setearla en el atributo data"
                warnings.warn(f"Warning........... {msg}")

            N = len(self.data) + 1
            num_slides, pos_division = fc.localmin(self.data, coef)
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
            self.slides = fc.select(sets, self.frames, 1, self.rgb, self.gpu_use) # 0 -> Se seleccionan los ultimos frames de cada conjunto 

        self.frames = fc.delete_frames(self.frames, self.slides, 1)

    def set_transcription(self, path = '', ocr = 1):
        """ Transcribe las imagenes  contenidas en self.frames, ordena la data y la deja almacenada en un array o json (dependiendo de self.runtime)
        -------------------------------------------------------
        Input:
            No aplica
        Output:
            No aplica
        """
        if (len(self.slides) == 0):
            self.set_slides()
            msg = "No se tienen las slides, se ejecuta automaticamente el metodo set_slides() para setearla en el atributo slides"
            warnings.warn(f"Warning........... {msg}")

        self.transcription = fc.get_transcription(self.video_name, self.frames, self.slides, self.rgb, self.runtime, self.gpu_use, path, ocr) # los dos casos cubiertos 

    def clean_frames(self, pix_lim = 0.001, ssimv_lim = 0.999): 
        """ Itera sobre los frames comparando usando metricas de calidad de imagen para eliminar las que sean consideradas suficientemente similares
        para el caso de no estar runtime : se elimina el frame de la ruta 
        caso runtime: se crea una nueva lista con los frames correspondientes y se retorna  
        -------------------------------------------------------
        Input:
            No aplica
        Output:
            No aplica
        """
        if(self.runtime):
            self.frames = fc.clean(self.frames, self.rgb, pix_lim, ssimv_lim)
        else:
            fc.clean(self.frames, self.rgb, pix_lim, ssimv_lim)

    def clean_transc(self):
        """  Desde una transcripcion formateada se eliminan redundancias y luego se eliminan los frames:
        runtime: se filtran sobre el array 
        no runtime: se eliminan los archivos de la ruta de los frames
        -------------------------------------------------------
        Input:
            No aplica
        Output:
            No aplica
        """
        self.transcription, lt_delet = fc.clean_transc(self.transcription)
        self.frames = fc.delete_frames(self.frames, lt_delet)

    def lematize(self):
        """  Funcion aplica lematizacion sobre la transcripcion almacenada en self.transcription, reemplazando la original
        -------------------------------------------------------
        Input:
            No aplica
        Output:
            No aplica
        """
        self.transcription = fc.lematize(self.transcription, self.gpu_use)

    def improve_num(self):
        """  Funcion corrige digitos de la transcripcion inicial, utilizando el OCR de Tesseract 
        -------------------------------------------------------
        Input:
            No aplica
        Output:
            No aplica
        """
        # tomar frames finales
        print("ENTRO")
        transcription_tesse = fc.get_transcription(self.video_name, self.frames, [], self.rgb, self.runtime, self.gpu_use, 2) # los dos casos cubiertos 
        print(transcription_tesse)

    def improve_quality(self, path, model, ratio):
        """  Funcion mejora calidad de imagenes, ya sea la lista de frames o frames guardados localmente
        -------------------------------------------------------
        Input:
            No aplica
            model
            ratio
        Output:
            No aplica
        """
        # self.frames
        if(self.runtime):
            for index, frame in enumerate(self.frames):
                self.frames[index] = fc.upscale_img(frame, path, model, ratio, self.runtime, self.gpu_use)
        else:
            Frames = fc.ls(ruta = self.frames)
            Frames.sort()
            Frames = list(map(fc.addJ ,Frames))
            for index, frame in enumerate(Frames):
                fc.upscale_img(self.frames+frame, path, model, ratio, self.runtime, self.gpu_use)



        # TODO: aplicar tesseract en los casos que se encuentre un digito o cifra en self.transcription (puede ir despues de lematization)
        # TODO: REVISAR SI EXISTE ALGUNA FORMA DE ENTREGAR MAYOR VALOR A LA ESTRUCTURACION ( ETIQUETAS ? : TITTLE, COMMENT, NAMES, NUMBER OR DATES)


        # TODO: REVISAR FORMAS DE OBTENER CONTEXTO DE INFO EN UNA LAMINA (QUIZAS FILTRAR Y OMITIR INFORMACION NO RELEVANTE)
        # TODO: N-GRAMA PARA LA CORRECCION -> REVISAR QUE PALABRAS SE REPITEN MAS Y QUIZAR HACER UNA ANALISIS ESTADISTICO CON ESTO (UN PLUS (?))
        # TODO: MENCIONAR QUE SE PUEDE MEJORAR EL CALCULO DE DISTANCIA ENTRE CUADRADOS DE TEXTO -> MEJORAR ESTRUCTURACION EN CASO DE TEXTO EN DIAGONAL
        # TODO: quizas quitar parametros desde la definicion de la clase y dejaros seteables luego de su creacion
        # TODO : dejar como parametro editable el limite para la limpieza por texto




        # DONE: Dejar con el formato de result de EASYOCR el resultado obtenido con tesseract 
        # DONE: MEJORAR FUNCION PARA ELEGIR FRAMES DESDE SLIDE (actual es last_one) -> AGREGAR A DOCUMENTO
        # DONE: REVISAR QUE TAN UTIL SERIA EL RTF (no mucho, se puede agregar formato, y quizas imagenes, -> buscar valor o uso de los RTF)
        
        # DONE: eleccion de lematizar
        # DONE dar libertad de los rangos a los cuales se desea filtrar
        # DONE revisar cambios necesarios para implementar solo escala de grises ( o dar la opcion de elegir) -> agregar comparacion al documento
