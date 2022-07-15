# Video Slides

Package made to obtain a text transcription from a video, with a flow from video to frames to structured frames to transcription in a json file 
Paquete hecho para obtener una transcripcion desde un video, con un flujo que inicia con un video se obtienen sus frames y finalmente se extrae la transcripcion estructurada 

## Instalacion:

Mediante el siguiente comando :

pip install videoslides

## Ejemplos de uso

    # Crear clase de Video

    video1 = Video(string, 100, 1, True)
    video1.clean_frames()
    video1.set_data()
    video1.set_slides() 
    video1.set_transcription()


    print(video1.data)
    print(video1.slides)
    print(video1.transcription)
    # ploteo(video1.video_name , video1.data) # grafica