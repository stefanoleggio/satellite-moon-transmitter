import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def iqPlot(values,title):
    plt.plot(np.real(values[:PLOT_POINTS]),np.imag(values[:PLOT_POINTS]), '.')
    plt.title(title)
    plt.xlabel("Q")
    plt.ylabel("I")
    plt.grid()
    plt.savefig("plots/"+title+".png")
    plt.close()

def waveContinuityPlot(output_signal):
    i = 1
    j = 2
            
    plt.scatter(np.arange(0,len(output_signal[i])), np.real(output_signal[i]))
    plt.scatter(np.arange( len(output_signal[i]),len(output_signal[i]) + len(output_signal[j])) , np.real(output_signal[j]))
    plt.axis([len(output_signal[i])-len(output_signal[i])/10, len(output_signal[j])+len(output_signal[i])/10, min(np.concatenate((output_signal[i], output_signal[2]))), max(np.concatenate((output_signal[i], output_signal[j])))])
    plt.show()

def quantize_uniform(x, quant_min=-1.0, quant_max=1.0, quant_level=5):

    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)-pow(2,BIT_FOR_IQ-1)

    return x_normalize_quant

def simulate_doppler_shift():

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

    return wave_list



if __name__ == '__main__':

    if(os.path.exists("output.bin")):
        os.remove("output.bin")

    message = open("message.bin", "rb")
    codes = json.load(open("codes.json", "r"))
    output_file = open("output.bin","ab")

    # PRN sequences already modulated with BOC

    BOC_SEQUENCE = codes['boc_sequence']
    BOC_SEQUENCE_INVERSE = codes['boc_sequence_inverse']
    
    # Chanel costants

    BOLTZMANN_COSTANT = 1.3809649 * pow(10,-23)
    TEMPERATURE = 300
    BAND = 1.023 * pow(10,6)
    N_0 = BOLTZMANN_COSTANT*TEMPERATURE
    F_S = 4.092*pow(10,6) # Sampling frequency
    V_FS = pow(10,-4)
    BIT_FOR_IQ = 16

    # Channel Parameters

    pathLossFlag = True
    awgnFlag = True 
    writeOutput = False

    # Doppler shift simulation

    ds_wave_list = simulate_doppler_shift()


    # Path loss simulation

    path_loss_vector = np.random.uniform(pow(10, -8),pow(10,-5),len(ds_wave_list))

    bit_counter = 0

    output_signal = [] # Output list of all signal waves

    while(True):

        print("Reading bit number " + str(bit_counter))
        
        message_bit = message.read(1)

        boc_sequence = []

        try:
            int(message_bit,2)
        except:
            break

        # PRN and BOC modulation

        if(int(message_bit,2)):
            boc_sequence = BOC_SEQUENCE_INVERSE
        else:
            boc_sequence = BOC_SEQUENCE

        # Doppler shift implementation

        repetitions = len(ds_wave_list[bit_counter])/(2*4092)

        repetitions_integer = math.modf(repetitions)[1]
        repetitions_decimal = math.modf(repetitions)[0]

        remainder = repetitions_decimal

        boc_output = []
        for i in range(len(boc_sequence)):
            if(remainder<1):
                j = 0
                while(j<repetitions_integer):
                    boc_output.append(boc_sequence[i])
                    j+=1
                remainder = remainder
            else:
                j = 0
                while(j<repetitions_integer+1):
                    boc_output.append(boc_sequence[i])
                    j+=1
                remainder = math.modf(remainder)[0]
            remainder += repetitions_decimal


        if(len(ds_wave_list[bit_counter]) != len(boc_output)):
            boc_output.append(boc_sequence[len(boc_sequence)-1])

        signal = boc_output * ds_wave_list[bit_counter]

        # AWGN simulation

        awgn_vector = (np.random.randn(len(boc_output)) + 1j*np.random.randn(len(boc_output))) * np.sqrt(N_0*BAND/2)

        if(pathLossFlag):
            
            # Apply Path Loss

            signal = signal*path_loss_vector[bit_counter]

        if(awgnFlag):

            # Apply AWGN

            signal = signal + awgn_vector

        output_signal.append(signal)

        if(writeOutput):

            # IQ samples writing

            for signal_bit in signal:
                real_sample = int(quantize_uniform(np.array([np.real(signal_bit)]), -V_FS, V_FS,pow(2,BIT_FOR_IQ))[0])
                imag_sample = int(quantize_uniform(np.imag([np.real(signal_bit)]), -V_FS, V_FS,pow(2,BIT_FOR_IQ))[0])
                output_file.write(real_sample.to_bytes(2,byteorder='big',signed=True))
                output_file.write(imag_sample.to_bytes(2,byteorder='big',signed=True))
                output_file.flush()

        bit_counter += 1
            

    waveContinuityPlot(output_signal)