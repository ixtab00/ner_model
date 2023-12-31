from preprocessing import DataLoader
import pipeline
from json import loads

def main():
    with open("./word_index.json") as file:
        windex = loads(file.read())

    with open("./tag_index.json") as file:
        tindex = loads(file.read())

    decoder_stage = pipeline.DecodingStage(tindex)
    encoder_stage = pipeline.EncodingStage(40, 50, windex, decoder_stage)
    main_stage = pipeline.ComputingStage('model_weights_cnn_mod.h5')

    pipe = pipeline.Pipeline(encoder_stage, main_stage, decoder_stage)
    
    while True:
        text = input("Text: ")
        print(pipe.pipeline(text))

main()