import json
import matplotlib.pyplot as plt
import numpy as np
import sys

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
    PRNCodes = json.load(open("PRNCodes.json", "r"))

    PRN_SEQUENCE = PRNCodes['code_sequence']
    PRN_SEQUENCE_INVERSE = PRNCodes['code_sequence_inverse']

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

        # Adding PRN

        chunk_PRN = []
        for message_bit in chunk:
            if(message_bit):
                chunk_PRN.append(PRN_SEQUENCE)
            else:
                chunk_PRN.append(PRN_SEQUENCE_INVERSE)

        print("Chunk with PRN")
        print(list(map(lambda x: x[:15] + "...", chunk_PRN)))

        print("\n")

        if(chunk_count == 0 and len(sys.argv)>1 and sys.argv[1] == "-p"):
            signalPlot(chunk,1,chunk_len, "Message")
            draw_plot = False
        
        chunk_count += 1
        bit_count = 0