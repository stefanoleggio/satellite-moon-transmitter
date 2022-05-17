import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

def signalPlot(data, T, total_duration, title):
    t = np.arange(0, total_duration, T)
    plt.step(t, data)
    plt.xticks(t)
    plt.title(title)
    plt.show()

 # Usage example:   python main.py
 #                  python main.py -p (for plotting)

if __name__ == '__main__':

    chunk_len = 10 # Bits for chunk - TODO: symbol rate as a variable multiple of the PRN duration

    message = open("message.bin", "rb")
    codes = json.load(open("PRNCodes.json", "r"))

    PRN_SEQUENCE = codes['prn_sequence']
    PRN_SEQUENCE_INVERSE = codes['prn_sequence_inverse']
    BOC_SEQUENCE = codes['boc_sequence']
    BOC_SEQUENCE_INVERSE = codes['boc_sequence_inverse']

    bit_count = 0
    chunk_count = 0
    eof = False

    while(not eof):

        chunk = []

        while(bit_count < chunk_len):

            message_bit = message.read(1)

            bit_count += 1

            # Splitting the message in chunks
            try:
                chunk.append(int(message_bit,2))
            except:
                eof=True
                break


        if(len(chunk) < 1):
            break
        
        print("Message chunk " + str(chunk_count))
        print(list(map(lambda x: bin(x)[2:], chunk)))

        print("\n")

        # Adding PRN

        chunk_PRN = []
        for message_bit in chunk:
            if(message_bit):
                chunk_PRN.append(PRN_SEQUENCE_INVERSE)
            else:
                chunk_PRN.append(PRN_SEQUENCE)

        print("Chunk with PRN")
        print(list(map(lambda x: x[:15] + "...", chunk_PRN)))

        print("\n")

        ## BOC Modulation

        chunk_boc = []
        for message_bit in chunk:
            if(message_bit):
                chunk_boc.append(BOC_SEQUENCE_INVERSE)
            else:
                chunk_boc.append(BOC_SEQUENCE)


        print("Chunk with BOC")
        print(list(map(lambda x: x[:15] + "...", chunk_boc)))

        print("\n")

        ## AWGN

        for message_bit in chunk_boc:
            awgn_vector = np.random.randn(len(message_bit)) + 1j * np.random.randn(len(message_bit))
            path_loss_vector = np.random.uniform(0,0.6,len(message_bit))
            message_bit_splitted = message_bit.split(" ")
            #message_bit_splitted_casted =[int(x) for x in message_bit_splitted]
            #print(message_bit_splitted[:10])
            #print(message_bit_splitted[0])

            #message_bit_splitted_casted = [int(x) for x in message_bit_splitted]

            #print(message_bit_splitted_casted)
            #print(message_bit_splitted)
            #print(path_loss_vector[:10])

        if(chunk_count == 0 and len(sys.argv)>1 and sys.argv[1] == "-p"):

            prn_to_plot = chunk_PRN[0][:chunk_len]
            prn_to_plot_list = []

            for char in prn_to_plot:
                prn_to_plot_list.append(int(char,2))
    
            signalPlot(chunk,1,chunk_len, "Message")
            signalPlot(prn_to_plot_list,1, chunk_len, "Message with PRN")
            #signalPlot(boc_to_plot_list,1, chunk_len, "Message modulated with BOC")
            draw_plot = False

        
        chunk_count += 1
        bit_count = 0