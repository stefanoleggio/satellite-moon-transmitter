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
    codes = json.load(open("codes.json", "r"))

    PRN_SEQUENCE = codes['prn_sequence']
    PRN_SEQUENCE_INVERSE = codes['prn_sequence_inverse']
    BOC_SEQUENCE = codes['boc_sequence']
    BOC_SEQUENCE_INVERSE = codes['boc_sequence_inverse']
    
    # Channel modelling
    BOLTZMANN_COSTANT = 1.3809649 * pow(10,-23)
    TEMPERATURE = 300
    BAND = 1.023 * pow(10,6)
    N_0 = BOLTZMANN_COSTANT*TEMPERATURE

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
        print(list(map(lambda x: str(x[:5]) + "...", chunk_PRN)))

        print("\n")

        # BOC Modulation

        chunk_boc = []
        for message_bit in chunk:
            if(message_bit):
                chunk_boc.append(BOC_SEQUENCE_INVERSE)
            else:
                chunk_boc.append(BOC_SEQUENCE)


        print("Chunk with BOC")
        print(list(map(lambda x: str(x[:5]) + "...", chunk_boc)))

        print("\n")

        # Adding channel effects

        chunk_channel = []

        for boc_values_message_bit in chunk_boc:

            boc_for_message_bit_size = len(boc_values_message_bit)

            awgn_vector = (np.random.randn(boc_for_message_bit_size) + 1j*np.random.randn(boc_for_message_bit_size)) * np.sqrt(N_0*BAND/2)
            path_loss_vector = np.random.uniform(0,pow(10, -8),boc_for_message_bit_size)
                     
            boc_values_message_bit_channel = boc_values_message_bit * path_loss_vector + awgn_vector

            """
            plt.plot(np.real(boc_values_message_bit_channel),np.imag(boc_values_message_bit_channel), '.')
            plt.grid()
            plt.show()
            """

            chunk_channel.append(boc_values_message_bit_channel)

        print("Chunk with channel effects: AWGN and PATH LOSS")
        print(chunk_channel)
        print("\n-------------------------------------------------\n")

        if(chunk_count == 0 and len(sys.argv)>1 and sys.argv[1] == "-p"):

            """
            prn_to_plot = chunk_PRN[0][:chunk_len]
            prn_to_plot_list = []

            for char in prn_to_plot:
                prn_to_plot_list.append(int(char,2))
    
            signalPlot(chunk,1,chunk_len, "Message")
            signalPlot(prn_to_plot_list,1, chunk_len, "Message with PRN")
            #signalPlot(boc_to_plot_list,1, chunk_len, "Message modulated with BOC")
            draw_plot = False
            """

        
        chunk_count += 1
        bit_count = 0