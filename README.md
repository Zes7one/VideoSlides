# Video Slides

Descripcion corta: 


## Instalacion:

Con el comando :
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