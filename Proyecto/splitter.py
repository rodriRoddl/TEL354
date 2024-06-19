from pydub import AudioSegment
import math

seg = 2

input_audio_file = "data-features/Benja2.wav" 
speech = AudioSegment.from_mp3(input_audio_file)

batch_size = seg * 1000
duracion = speech.duration_seconds
batches = math.ceil(duracion / seg)

inicio = 0
for i in range(batches):
    pedazo = speech[inicio: inicio + batch_size]
    pedazo.export(f'pedazo_{i}.wav', format='wav') #guardamos el audio
    inicio+= batch_size