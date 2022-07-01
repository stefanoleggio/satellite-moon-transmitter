from cmath import phase
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle

def iqPlot(values,title):
    plt.plot(np.real(values[:PLOT_POINTS]),np.imag(values[:PLOT_POINTS]), '.')
    plt.title(title)
    plt.xlabel("Q")
    plt.ylabel("I")
    plt.grid()
    plt.savefig("plots/"+title+".png")
    plt.close()


def quantize_uniform(x, quant_min=-1.0, quant_max=1.0, quant_level=5):

    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)-pow(2,BIT_FOR_IQ-1)

    return x_normalize_quant

def quantizer(value):

    quantized_value = abs(round(value/V_FS * (pow(2,BIT_FOR_IQ-1)-1)))
    if(abs(value) > V_FS):
        quantized_value = pow(2,BIT_FOR_IQ - 1) - 1
    return int(math.copysign(1,value)) * quantized_value

 # Usage example:   python main.py
 #                  python main.py -p (for plotting)

if __name__ == '__main__':

    chunk_len = 90 # Bits for chunk - TODO: symbol rate as a variable multiple of the PRN duration

    if(os.path.exists("output.bin")):
        os.remove("output.bin")

    message = open("message.bin", "rb")
    codes = json.load(open("codes.json", "r"))
    output_file = open("output.bin","ab")

    PRN_SEQUENCE = codes['prn_sequence']
    PRN_SEQUENCE_INVERSE = codes['prn_sequence_inverse']
    BOC_SEQUENCE = codes['boc_sequence']
    BOC_SEQUENCE_INVERSE = codes['boc_sequence_inverse']
    
    # Channel modelling
    BOLTZMANN_COSTANT = 1.3809649 * pow(10,-23)
    TEMPERATURE = 300
    BAND = 1.023 * pow(10,6)
    N_0 = BOLTZMANN_COSTANT*TEMPERATURE
    F_S = 4.092*pow(10,6) # Sampling frequency


    V_FS = pow(10,-4)
    BIT_FOR_IQ = 16

    PLOT_POINTS = 300

    bit_count = 0
    chunk_count = 0
    eof = False
    noiseFlag = True # Flag for enabling awgn noise
    firstPlot = True

    #-----------------------

        # Simulation of doppler shift

    doppler_shift_vector = np.random.uniform(2000,5000,90) #TODO: segni negativi?

    iq_duration_doppler = 1/(BAND+doppler_shift_vector)

    phases_shifts_vector = []

    original_bit_duartion = 4092*2*iq_duration_doppler

    time_list = []


    # Phase shift calculation
    for i in range(len(original_bit_duartion)):
        time_list.append(np.arange(0, original_bit_duartion[i], 1/F_S))
        if(i>0):
            phases_shifts_vector.append(2*math.pi*doppler_shift_vector[i-1]*original_bit_duartion[i-1] + phases_shifts_vector[i-1])
        else:
            phases_shifts_vector.append(0)

    # Wave generation
    wave_list = []
    for i in range(len(time_list)):
        wave_list.append(np.cos(2*math.pi*doppler_shift_vector[i]*time_list[i] + phases_shifts_vector[i])) #TODO: change cos

    #-----------------------

    #Path loss
    path_loss_vector = np.random.uniform(pow(10, -8),pow(10,-5),len(phases_shifts_vector))

   
    bit_counter = 0

    output_signal = []

    while(not eof):

        print("Reading bit number " + str(bit_counter))
        
        message_bit = message.read(1)

        boc_sequence = []

        try:
            int(message_bit,2)
        except:
            eof = True
            break

        if(int(message_bit,2)):
            boc_sequence = BOC_SEQUENCE_INVERSE
        else:
            boc_sequence = BOC_SEQUENCE

        repetitions = len(wave_list[bit_counter])/(2*4092)

        repetitions_integer = math.modf(repetitions)[1]
        repetitions_decimal = math.modf(repetitions)[0]

        resto = repetitions_decimal

        boc_output = []
        for i in range(len(boc_sequence)):
            if(resto<1):
                j = 0
                while(j<repetitions_integer):
                    boc_output.append(boc_sequence[i])
                    j+=1
                resto = resto
            else:
                j = 0
                while(j<repetitions_integer+1):
                    boc_output.append(boc_sequence[i])
                    j+=1
                resto = math.modf(resto)[0]
            resto += repetitions_decimal


        if(len(wave_list[bit_counter]) != len(boc_output)):
            boc_output.append(boc_sequence[len(boc_sequence)-1])

        signal = boc_output * wave_list[bit_counter]

        awgn_vector = (np.random.randn(len(boc_output)) + 1j*np.random.randn(len(boc_output))) * np.sqrt(N_0*BAND/2)

        signal = signal*path_loss_vector[bit_counter]

        if(noiseFlag):
            signal = signal + awgn_vector

        output_signal.append(signal)

        bit_counter += 1
        """
        for boc_value_message_bit_channel in signal:
            real_sample = int(quantize_uniform(np.array([np.real(boc_value_message_bit_channel)]), -V_FS, V_FS,pow(2,BIT_FOR_IQ))[0])
            imag_sample = int(quantize_uniform(np.imag([np.real(boc_value_message_bit_channel)]), -V_FS, V_FS,pow(2,BIT_FOR_IQ))[0])
            output_file.write(real_sample.to_bytes(2,byteorder='big',signed=True))
            output_file.write(imag_sample.to_bytes(2,byteorder='big',signed=True))
            output_file.flush()
        """

    #plt.scatter(np.arange(0,len(output_signal[1])), np.real(output_signal[1]))

    #plt.scatter(np.arange( len(output_signal[1]),len(output_signal[1]) + len(output_signal[2])) , np.real(output_signal[2]))

    #plt.show()




