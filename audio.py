import cv2
# import numpy as np
# ffpyplayer for playing audio
# from ffpyplayer.player import MediaPlayer


from scipy import fft, arange
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

import subprocess
import moviepy.editor as mp
from pydub import AudioSegment
import moviepy.editor as mp
import scipy.fftpack
# def PlayVideo(video_path):
#     video=cv2.VideoCapture(video_path)
#     player = MediaPlayer(video_path)
#     while True:
#         grabbed, frame=video.read()
#         audio_frame, val = player.get_frame()
#         if not grabbed:
#             print("End of video")
#             break
#         if cv2.waitKey(28) & 0xFF == ord("q"):
#             break
#         cv2.imshow("Video", frame)
#         if val != 'eof' and audio_frame is not None:
#             #audio
#             img, t = audio_frame
#     video.release()
#     cv2.destroyAllWindows()


# video_path="../L1/images/Godwin.mp4"
# RC2 = "./video2/y2mate.com - VESKI_360p.mp4"
# PlayVideo(RC2)


def mp4_mp3(ruta):
    #Cargamos el fichero .mp4
    clip = mp.VideoFileClip(ruta)

    #Lo escribimos como audio y `.mp3`
    clip.audio.write_audiofile("transformado_a.mp3")



def mp3omp4_wav(rutao, rutad):
    # command = "ffmpeg -i "+ruta+" -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    # subprocess.call(command, shell=True)
    clip = mp.VideoFileClip(rutao) #.subclip(0,20)
    # clip.audio.write_audiofile("theaudio.wav")
    clip.audio.write_audiofile(rutad)
    # print(rutao)
    # sound = AudioSegment.from_mp3(rutao)
    # sound.export(rutad, format="wav")

    


def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)



def graph_audio(rutao):
    nombre = rutao.split("\\")[-1]
    here_path = os.path.dirname(os.path.realpath(__file__))
    file_wav = 'audio.wav'
    audio_wav = os.path.join(here_path, file_wav)
    mp3omp4_wav(rutao, audio_wav)
    sr, signal = wavfile.read(audio_wav)
    os.remove(audio_wav)    

    np.set_printoptions(threshold=np.inf)
    N = signal.shape[0]
    print(type(signal))
    print(signal.shape)
    # exit(1)
    y = signal[:, 0]  # use the first channel (or take their average, alternatively)
    yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
    t = np.arange(len(y)) / float(sr)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')

    # frq, X = frequency_spectrum(y, sr)

    plt.subplot(3, 1, 2)
    plt.plot(t, yhat, 'b')
    plt.xlabel('t')
    plt.ylabel('y smooth')
    plt.tight_layout()
    # plt.title(nombre)
    # plt.show()

    # segundo intento de soften

    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, t[1]-t[0])
    spectrum = w**2
    cutoff_idx = spectrum < (spectrum.max()/5)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)

    plt.subplot(3, 1, 3)
    plt.plot(t, y2, 'b')
    plt.xlabel('t')
    plt.ylabel('y smooth')
    plt.tight_layout()
    plt.title(nombre)
    plt.show()


RC   = "video2\\Data Health.mp4"
RC2  = "video2\\VESKI_360p.mp4"
RC3  = "video2\\Estrategia Digital MBA UC examen_360p.mp4"
RC4  = "video2\\Un alivio a un click de distancia_360p.mp4"
RC5  = "video2\\MBAUC  Q22021  Estrategia Digital  Grow  Invest_360p.mp4"
RC6  = "video2\\PLATAFOMRA DE SEGUROSEST DIGITAL_360p.mp4"
RC7  = "video2\\Pitch Lifetech_360p.mp4"
RC8  = "video2\\Presentacion   TRADE NOW_360p.mp4.webm"
RC9  = "video2\\Closet Cleanup_360p.mp4.webm"
RC10 = "video2\\Almacén Digital_360p.mp4"


here_path = os.path.dirname(os.path.realpath(__file__))
test_mp4 = 'video2\\test.mp4'
# test = os.path.join(here_path, test_mp4)
# graph_audio(test)

graph_audio(os.path.join(here_path, RC))
graph_audio(os.path.join(here_path, RC2))
graph_audio(os.path.join(here_path, RC3))
graph_audio(os.path.join(here_path, RC4))
graph_audio(os.path.join(here_path, RC5))
graph_audio(os.path.join(here_path, RC6))
graph_audio(os.path.join(here_path, RC7))
graph_audio(os.path.join(here_path, RC8))
graph_audio(os.path.join(here_path, RC9))
graph_audio(os.path.join(here_path, RC10))