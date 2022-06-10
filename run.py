import io
import os
# from google.cloud import vision
import cv2
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pathlib import Path
import re
import time
import easyocr

# import cv2
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
	

def vid2text(RutaCompleta):

	reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

	# VideoName = "Data Health.mp4"
	# RutaVideo = "./video2/"
	inicio = time.time()
	VideoName = RutaCompleta.split("/")[-1]
	RutaVideo = RutaCompleta.replace(VideoName, '')
	print("----------------------------------------")
	print("VideoName : |"+VideoName+"|")
	print("RutaVideo : |"+RutaVideo+"|")

	# rutaFrames = RutaVideo+"Frames"
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
	print("FPS -> " , fps)
	print("Numero de frames: ",length)

	count = 0
	success,image = vidcap.read()
	while (count <= length):
		if(count%(fps*2) == 0):
			cv2.imwrite(RutaVideo+"%d.jpg" % count, image)     # save frame as JPEG file  
		success,image = vidcap.read()
		count += 1


	return "FIN"

	# ------- Eliminando Frames con info redundante -------
	# Lista de Frames
	def ls(ruta = Path.cwd()):
		return [int(arch.name.split(".")[0]) for arch in Path(ruta).iterdir() if (arch.is_file() and re.search(r'\.jpg$', arch.name))]
	Frames = ls(ruta = RutaVideo)
	Frames.sort()
	#definicion de variables
	height, width, channels = 0, 0 , 0
	pix_num = 0
	anterior = 0
	# iteracion sobre Frames contiguos, comparando por cantidad de pixeles diferntes y por metric SSIM, para eliminar los con info repetida
	for index, i in enumerate(Frames):
		im = cv2.imread(RutaVideo+str(i)+".jpg")
		if (index  == 0):
			# Numero de pixeles por frame
			height, width, channels = im.shape
			pix_num = height * width * 3
			im2 = im.copy()
			img2 = im.copy()
			anterior = i
			continue

		img1 = img_as_float(img2)
		img2 = img_as_float(im)
		im1 = im2
		im2 = im 
		# Aplicando metrica SSIM
		ssimV = ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min())
		# total number of different pixels between im1 and im2
		dif = np.sum(im1 != im2)
		#print(anterior," -> ",i)
		if (False):
			if ( dif/pix_num < 0.5):
				#  Son escencial- la misma
				direc = RutaVideo+str(anterior)+".jpg"
				os.remove(direc)
			elif(ssimV>0.85):			
					# Son escencial- la misma
					direc = RutaVideo+str(anterior)+".jpg"
					os.remove(direc)
		anterior = i

	transcription = "" 	
	# if (True):
	# 	return transcription
	# ------- Extrayendo texto desde Frames filtrados -------

	# seteando credenciales para uso de VISION 
	# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "eminent-crane-329018-d010213b1439.json"
	# cliente = vision.ImageAnnotatorClient()

	Frames = ls(ruta = RutaVideo)
	Frames.sort()

	# se itera sobre Frames ya filtrados y se hacen las consultas en la API Vision
	for index, i in enumerate(Frames):
		#lectura del frame 
		direc = RutaVideo+str(i)+'.jpg'

		# result = reader.readtext(invert, detail = 0) # derecho
		# result = reader.readtext(direc, detail = 0) # derecho
		# transcription = transcription+">> "+(" ").join(result)+"\n"
		# os.remove(direc)

	# os.rmdir(RutaVideo)

	
	VideoName = VideoName+".txt"
	with open(VideoName, "w", encoding='utf-8') as text_file:
		text_file.write(transcription)
	
	fin = time.time()
	print("TIME : %d [seg]" % round(fin-inicio, 2)) 
	print("----------------------------------------")
	return transcription





RC = "./video2/Data Health.mp4"
RC2 = "./video2/y2mate.com - VESKI_360p.mp4"
RC3 = "./video2/y2mate.com - Estrategia Digital MBA UC examen_360p.mp4"
RC4 = "./video2/y2mate.com - Un alivio a un click de distancia_360p.mp4"
RC5 = "./video2/y2mate.com - MBAUC  Q22021  Estrategia Digital  Grow  Invest_360p.mp4"
RC6 = "./video2/y2mate.com - PLATAFOMRA DE SEGUROSEST DIGITAL_360p.mp4"
RC7 = "./video2/y2mate.com - Pitch Lifetech_360p.mp4"
RC8 = "./video2/y2mate.com - Presentacion   TRADE NOW_360p.mp4.webm"
RC9 = "./video2/y2mate.com - Closet Cleanup_360p.mp4.webm"
RC10 = "./video2/y2mate.com - Almacén Digital_360p.mp4"



# print(RC)
print(vid2text(RC2))