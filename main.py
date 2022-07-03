import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

def quantize_uniform(x, quant_min=-1.0, quant_max=1.0, quant_level=5):

    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)-pow(2,BIT_FOR_IQ-1)

    return x_normalize_quant

def simulate_doppler_shift(ds_duration, input_bits, freq_min, freq_max):

    total_points = ds_duration * F_S

    doppler_shift_vector = []

    max_length = input_bits * 4092 * 2 /(2*BAND-freq_max) * F_S

    doppler_shift_vector = np.random.uniform(freq_min,freq_max,math.ceil(max_length/total_points))


    phases_shifts_vector = []

    time_list = np.arange(0, ds_duration, 1/F_S)

    # Phase shift calculation
    for i in range(len(doppler_shift_vector)):
        if(i>0):
            phases_shifts_vector.append(2*math.pi*doppler_shift_vector[i-1]*ds_duration + phases_shifts_vector[i-1])
        else:
            phases_shifts_vector.append(0)


    # Wave generation
    wave_list = []
    for i in range(len(doppler_shift_vector)):
        wave_list = np.concatenate((wave_list,np.cos(2*math.pi*doppler_shift_vector[i]*time_list + phases_shifts_vector[i]) + 1j * np.sin(2*math.pi*doppler_shift_vector[i]*time_list + phases_shifts_vector[i])))
    

    return wave_list, doppler_shift_vector


def square_wave_plot(data, T, total_duration, title):
    t = np.arange(0, total_duration, T)
    plt.step(t, data)
    plt.xlabel("n (sample)")
    plt.ylabel("value")
    plt.xticks(t)
    plt.title(title)
    plt.savefig("plots/"+title+".png")
    plt.close()


def simulate_path_loss(pl_duration, input_bits, val_min, val_max, freq_max):

    total_points = pl_duration * F_S


    path_loss_vector = []

    j = 0

    max_length = input_bits * 4092 * 2 /(2*BAND-freq_max) * F_S

    while(j<max_length):
        x = np.random.uniform(val_min,val_max,1)[0]
        i = 0
        while(i<total_points):
            path_loss_vector.append(x)
            i += 1

        j += i

    return path_loss_vector

if __name__ == '__main__':

    if(os.path.exists("output.bin")):
        os.remove("output.bin")

    message = open("message.bin", "rb")
    codes = json.load(open("codes.json", "r"))
    output_file = open("output.bin","ab")

    # PRN sequences already modulated with BOC

    BOC_SEQUENCE = codes['boc_sequence']
    BOC_SEQUENCE_INVERSE = codes['boc_sequence_inverse']
    PRN_SEQUENCE = codes['prn_sequence']
    PRN_SEQUENCE_INVERSE = codes['prn_sequence_inverse']
    
    # Chanel costants

    BOLTZMANN_COSTANT = 1.3809649 * pow(10,-23)
    TEMPERATURE = 300
    BAND = 1.023 * pow(10,6)
    N_0 = BOLTZMANN_COSTANT*TEMPERATURE
    F_S = 4.092*pow(10,6) # Sampling frequency
    V_FS = pow(10,-4)
    BIT_FOR_IQ = 16

    # Channel Parameters

    ds_duration_default = 0.2
    pl_duration_default = 0.1995
    ds_freq_max_default = 5000
    ds_freq_min_default = 2000
    pl_val_min_default = -6
    pl_val_max_default = -5

    print("\n### Satellite transmitter simulator ###\n")

    ds_duration = float(input("Set Doppler shift duration (" + str(ds_duration_default) + "): ") or ds_duration_default)

    pl_duration = float(input("Set Path Loss duration: (" + str(pl_duration_default) + "): ") or pl_duration_default)

    ds_freq_max = int(input("Set Doppler shift max freq: (" + str(ds_freq_max_default) + "): ") or ds_freq_max_default)

    ds_freq_min = int(input("Set Doppler shift min freq: (" + str(ds_freq_min_default) + "): ") or ds_freq_min_default)

    pl_val_min = int(input("Set Path Loss min value: (" + str(pl_val_min_default) + "): ") or pl_val_min_default)

    pl_val_max = int(input("Set Path Loss max value: (" + str(pl_val_max_default) + "): ") or pl_val_max_default)

    pathLossFlag = input("Insert Path Loss? (Y/N)")

    if(pathLossFlag.lower() == "y"):
        pathLossFlag = True
    elif(pathLossFlag.lower() == "n"):
        pathLossFlag = False
    else:
        pathLossFlag = True

    awgnFlag = input("Insert AWGN? (Y/N)")

    if(awgnFlag.lower() == "y"):
        awgnFlag = True
    elif(awgnFlag.lower() == "n"):
        awgnFlag = False
    else:
        awgnFlag = True

    writeOutput = input("Write IQ samples output? (Y/N)")

    if(writeOutput.lower() == "y"):
        writeOutput = True
    elif(writeOutput.lower() == "n"):
        writeOutput = False
    else:
        writeOutput = False


    ds_wave_list, doppler_shift_vector = simulate_doppler_shift(ds_duration, 180, ds_freq_min, ds_freq_max)

    path_loss_vector = simulate_path_loss(pl_duration, 180, pow(10, pl_val_min), pow(10,pl_val_max), ds_freq_max)

    bit_counter = 0

    boc_output = []

    current_time = 0
    remainder = 0

    message_bits_to_plot = [] # Buffer for plotting
    messagePlotFlag = True
    bocPlotFlag = True
    prnPlotFlag = True

    while(True):
        
        message_bit = message.read(1)

        message_bits_to_plot.append(message_bit)

        if(len(message_bits_to_plot)>10 and messagePlotFlag):
            square_wave_plot(message_bits_to_plot, 1, len(message_bits_to_plot), "Message")
            messagePlotFlag = False

        boc_sequence = []

        try:
            int(message_bit,2)
        except:
            break

        # Just for plotting: PRN adding
        PRN_sequence = []
        if(int(message_bit,2)):
            PRN_sequence = PRN_SEQUENCE_INVERSE
        else:
            PRN_sequence = PRN_SEQUENCE

        if(prnPlotFlag):
            square_wave_plot(PRN_sequence[:10],1,10,"Message with PRN")
            prnPlotFlag = False

        # PRN and BOC modulation, the BOC(1,1) sequence already contains the PRN

        if(int(message_bit,2)):
            boc_sequence = BOC_SEQUENCE_INVERSE
        else:
            boc_sequence = BOC_SEQUENCE

        if(bocPlotFlag):
            square_wave_plot(boc_sequence[:10],1,10,"Message modulated with Boc(1,1)")
            bocPlotFlag = False

        # Doppler shift implementation

        repetitions = []

        for i in range(len(boc_sequence)):
            index = math.floor(current_time/ds_duration)
            current_time += 1/(2*BAND+doppler_shift_vector[index])
            repetitions.append(1/(2*BAND+doppler_shift_vector[index])*F_S)

        

        for i in range(len(boc_sequence)):
            remainder += repetitions[i]

            j = 0
            while(j<math.modf(remainder)[1]):
                boc_output.append(boc_sequence[i])
                j+=1
            remainder = math.modf(remainder)[0]


        bit_counter += 1


    signal = boc_output * ds_wave_list[:len(boc_output)]


    plt.scatter(np.arange(0,len(signal[815000:820000])), np.real(signal[815000:820000]))
    plt.title("Signal with Doppler Shift")
    plt.show()

    # AWGN simulation

    awgn_vector = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * np.sqrt(N_0*BAND/2)


    if(pathLossFlag):

        # Apply Path Loss

        signal = signal * path_loss_vector[:len(signal)]
    
    if(awgnFlag):

        # Apply AWGN

        signal = signal + awgn_vector

    plt.scatter(np.arange(0,len(signal[814000:820000])), np.real(signal[814000:820000]))
    plt.title("Signal with Path Loss and AWGN")
    plt.show()

    if(writeOutput):

        print("\nWriting output...")

        # IQ samples writing

        for signal_bit in signal:
            real_sample = int(quantize_uniform(np.array([np.real(signal_bit)]), -V_FS, V_FS,pow(2,BIT_FOR_IQ))[0])
            imag_sample = int(quantize_uniform(np.imag([np.real(signal_bit)]), -V_FS, V_FS,pow(2,BIT_FOR_IQ))[0])
            output_file.write(real_sample.to_bytes(2,byteorder='big',signed=True))
            output_file.write(imag_sample.to_bytes(2,byteorder='big',signed=True))
            output_file.flush()