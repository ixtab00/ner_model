from load_data import DataLoader
import pipeline
from json import loads

def main():
    with open("./word_index.json") as file:
        windex = loads(file.read())

    with open("./tag_index.json") as file:
        tindex = loads(file.read())

    decoder_stage = pipeline.PSDecode(tindex)
    encoder_stage = pipeline.PSEncode(windex, decoder_stage)
    main_stage = pipeline.PSCompute('./weights/model_weights(1).h5')

    pipe = pipeline.Pipeline(encoder_stage, main_stage, decoder_stage)
    
    while True:
        text = input("Text: ")
        print(pipe.pipeline(text))

main()