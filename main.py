import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def iqPlot(values,title):
    plt.plot(np.real(values),np.imag(values), '.')
    plt.title(title)
    plt.grid()
    plt.show()


def quantizer(value):

    quantized_value = abs(round(value/V_FS * (pow(2,BIT_FOR_IQ-1)-1)))
    if(abs(value) > V_FS):
        quantized_value = pow(2,BIT_FOR_IQ - 1) - 1
    
    if(value<0):
        quantized_value = ~quantized_value + pow(2,BIT_FOR_IQ-1) + 1
    return quantized_value

 # Usage example:   python main.py
 #                  python main.py -p (for plotting)

if __name__ == '__main__':

    chunk_len = 10 # Bits for chunk - TODO: symbol rate as a variable multiple of the PRN duration

    if(os.path.exists("output.bin")):
        os.remove("output.bin")

    message = open("message.bin", "rb")
    codes = json.load(open("codes.json", "r"))
    output_file = open("output.bin","a")

    PRN_SEQUENCE = codes['prn_sequence']
    PRN_SEQUENCE_INVERSE = codes['prn_sequence_inverse']
    BOC_SEQUENCE = codes['boc_sequence']
    BOC_SEQUENCE_INVERSE = codes['boc_sequence_inverse']
    
    # Channel modelling
    BOLTZMANN_COSTANT = 1.3809649 * pow(10,-23)
    TEMPERATURE = 300
    BAND = 1.023 * pow(10,6)
    N_0 = BOLTZMANN_COSTANT*TEMPERATURE
    F_S = 100 # Sampling frequency

    V_FS = pow(10,-4)
    BIT_FOR_IQ = 16

    bit_count = 0
    chunk_count = 0
    eof = False
    noiseFlag = True # Flag for enabling awgn noise
    firstPlot = True

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


            # Parameters for simulation

            awgn_vector = (np.random.randn(boc_for_message_bit_size) + 1j*np.random.randn(boc_for_message_bit_size)) * np.sqrt(N_0*BAND/2)
            path_loss_vector = np.random.uniform(pow(10, -8),pow(10,-5),boc_for_message_bit_size)
            doppler_shift_vector = np.random.uniform(2000,5000,boc_for_message_bit_size)
            latency_vector = np.random.uniform(5,10,boc_for_message_bit_size)*pow(10,-6)

            if(firstPlot and len(sys.argv)>1 and sys.argv[1] == "-p"):
                iqPlot(boc_values_message_bit, "BOC(1,1) output")

            boc_values_message_bit_channel = boc_values_message_bit * path_loss_vector

            if(firstPlot and len(sys.argv)>1 and sys.argv[1] == "-p"):
                iqPlot(boc_values_message_bit_channel, "IQ samples with path loss")

            boc_values_message_bit_channel = boc_values_message_bit_channel * np.exp(1j*2*np.pi*doppler_shift_vector*latency_vector)

            if(firstPlot and len(sys.argv)>1 and sys.argv[1] == "-p"):
                iqPlot(boc_values_message_bit_channel, "IQ samples with doppler shift")

            if(noiseFlag):
                boc_values_message_bit_channel = boc_values_message_bit_channel + awgn_vector
                if(firstPlot and len(sys.argv)>1 and sys.argv[1] == "-p"):
                    iqPlot(boc_values_message_bit_channel, "IQ samples with AWGN")


            for boc_value_message_bit_channel in boc_values_message_bit_channel:
                real_sample = quantizer(np.real(boc_value_message_bit_channel))
                imag_sample = quantizer(np.imag(boc_value_message_bit_channel))
                output_file.write(bin(real_sample)[2:].zfill(16))
                output_file.write(bin(imag_sample)[2:].zfill(16))

            chunk_channel.append(boc_values_message_bit_channel)

            firstPlot = False

        print("Chunk with channel effects: AWGN, PATH LOSS and DOPPLER SHIFT")
        print(chunk_channel)
        print("\n-------------------------------------------------\n")
        
        chunk_count += 1
        bit_count = 0