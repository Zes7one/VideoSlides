# !pip install easyocr
# !pip uninstall opencv-python-headless
# !pip install opencv-python-headless==4.1.2.30

import easyocr
import time
import cv2
import pytesseract 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
# from matplotlib.transforms import Affine2D
# import mpl_toolkits.axisartist.floating_axes as floating_axes
from autocorrect import Speller
import math
from PIL import Image
import enchant
import numpy as np

def tese(ruta, debug = False):
	inicio = time.time()
	pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
	# Grayscale, Gaussian blur, Otsu's threshold
	image = cv2.imread(ruta, 0)

	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(gray, (3,3), 0)
	# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

	# # Morph open to remove noise and invert image
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
	# invert = 255 - opening

	# Perform text extraction~
	# --psm N
	# 1 = Automatic page segmentation with OSD.
	# 3 = Fully automatic page segmentation, but no OSD. (Default)
	# 4 = Assume a single column of text of variable sizes.
	# 5 = Assume a single uniform block of vertically aligned text.
	# 6 = Assume a single uniform block of text.
	# 11 = Sparse text. Find as much text as possible in no particular order.
	# 12 = Sparse text with OSD.
	conf = f'--psm 6'
	data = pytesseract.image_to_string(image, lang='eng', config=conf)
	# show frame used
	if (debug == True):
		# plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
		# plt.imshow(opening, cmap = 'gray', interpolation = 'bicubic')
		plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()
		fin = time.time()
		print("TIME : %d [seg]" % round(fin-inicio, 2)) 

		# mejores_psm = [1, 3 , 4, 6, 11, 12] # 5, 
		# mejores = [0,1,2,3]
		# for i in mejores:
		#     conf = f'--psm 6 --oem {i}'
		#     # print(conf)
		#     try:
		#         data = pytesseract.image_to_string(image, lang='eng', config=conf)
		#         # data = pytesseract.image_to_string(image, lang='spa', config=conf)
		#         print(f"#########################  {i}  ###############################")
		#         print(data)
		#         print("############################################################")
		#     except Exception as e:
		#         print(e)

	return data


def easy(ruta, detail):
	
	reader = easyocr.Reader(['en'], gpu=False) # this needs to run only once to load the model into memory
	result = reader.readtext(ruta, detail = detail)
	if (detail == 1):
		if(isinstance(ruta, str)):
			im = cv2.imread(ruta)
		else:
			im = ruta
		# Create figure and axes
		fig_dims = (5, 5)
		fig, ax = plt.subplots(figsize=fig_dims)
		# Display the image
		ax.imshow(im)
		ejex = 0
		ejey = 0
		trans = ""
		for  pos, text, accu in result :			
			trans = trans + text + "\n"
			if ( pos[2][0] > ejex): 
				ejex = pos[2][0] 
			if ( pos[2][1] > ejey): 
				ejey = pos[2][1] 
			ancho = pos[1][0] - pos[0][0]
			alto = pos[2][1] - pos[1][1]
			x, y =  pos[0]
			# Create a Rectangle patch
			rect = patches.Rectangle((x, y), ancho, alto, linewidth=1, edgecolor='r', facecolor='none')
			# Add the patch to the Axes
			ax.add_patch(rect)
		ax.set_xlim(0, ejex+10)
		ax.set_ylim(0, ejey+10)
		ax.invert_yaxis()
		print(trans)
		plt.show()
		# plt.grid(True)
		# fin = time.time()
		# print("TIME : %d [seg]" % round(fin-inicio, 2)) 
		return trans
	return (" ").join(result)


def spell_check(text):
	"""
	Input: String con texto a corregir
	Output: String con texto corregido
	"""
	out = []
	spell = Speller(lang='es')
	text = text.split(' ')
	for i in range(len(text)):
		out.append(spell(text[i]))
	return ' '.join(text)

print(" ....................  INICIO ....................\n ")

# broker = enchant.Broker()
# broker.describe()
# print(broker.list_languages())
# print(broker.list_dicts())
# print(enchant.list_languages())
# help(enchant)
# exit(1)

def min_dis_sq(pos1, pos2): # arrays con coordenadas de ambos cuadrados
	a1,a2,a3,a4 = pos1
	b1,b2,b3,b4 = pos2
	#  usar puntos 1 y 3, ya que son de lados iguales
	if (a3[0] < b1[0]): # B esta completamente a la derecha de A -> A<<B 
		print("CASO 1")
		if (a1[1] > b3[1]):
			print("A")
		elif (b1[1] > a3[1]):
			print("G")
		elif ( b1[1] <= a1[1] <= b3[1] or  b1[1] <= a3[1] <= b3[1]):
			print(" B, C, D, E, F")


	elif (b3[0] < a1[0]):
		print("CASO 7")
		if (a1[1] > b3[1]):
			print("A")
		elif (b1[1] > a3[1]):
			print("G")
		elif ( b1[1] <= a1[1] <= b3[1] or  b1[1] <= a3[1] <= b3[1]):
			print(" B, C, D, E, F")


	elif ( b1[0] <= a1[0] <= b3[0] or  b1[0] <= a3[0] <= b3[0]):
		print("2, 3, 4, 5, 6")
		if (a1[1] > b3[1]):
			print("A")
		elif (b1[1] > a3[1]):
			print("G")
		else:
			print("resto")

	else:
		print("FALLO")

		

	# ancho = pos[1][0] - pos[0][0]
	# alto = pos[2][1] - pos[1][1]
	




	# Si se sobrelapan :  cuando 
	# Si a esta sobre b - > 
	# Si a esta a la derecha de b



	return 2


frame  = "4800.jpg"
frame  = "bn.png"
frame  = "232.jpg"
frame  = "348.jpg" 
frame  = "6148.jpg" 
frame  = "test/1.jpg" 
# frame  = "test/2.jpg" 
ruta  = '../CLEAN/clean_EstrategiaDigitalMBAUCexamen_360p/'+frame
img1 = cv2.imread(ruta, 0)
# img1 = plt.imread(ruta, 'RGB' )

# ------------------- Cambio de Escala -------------------
# width, height = 1024, 288
# width, height = 640, 360
# scale = 2
# width, height = (math.ceil(width*scale), math.ceil(height*scale))
# print(width, height)
# dim = (width, height)
# image = cv2.imread(ruta)
# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# # image = Image.open(ruta)
# # resized = image.resize(dim)
# cv2.imwrite(ruta, resized)  


# print(" .......... tese .......... ")
# tra_tese = tese(ruta)
# print(tra_tese)
# print(" .......... tese .......... ")
# exit(1)

print(" .......... easy .......... ")
tra_easy = easy(ruta, 1)
exit(1)
print(type(img1))
print(img1.shape)
# img2 = img1.ravel()
# print(type(img1))
tra_easy = easy(img1, 1)
# print(tra_easy)
print(" .......... easy .......... ")
test = "Hol muy buenos das, un guste en conocerlw el placwr es mip"

print(spell_check(test))

# result = tra_tese.split(' ')
# result = tra_easy.split(' ')
# out = []
# print(result)
# spell = Speller(lang='es')
# for i in range(len(result)):
#     # print(result[i], spell(result[i]))
#     out.append(spell(result[i]))
#     # result[i] = spell(result[i])
# print(' '.join(result))
# print(" .................... FIN .................... ")
# print(' '.join(out))








	   