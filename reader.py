
#Read the first 32 bit of the output.bin file

output_file = open("output.bin","rb")

counter = 0

print("Print of the first 32 IQ samples of the file output.bin")

while True:
    if(counter>32):
        break
    real_sample = output_file.read(2)
    real_sample_cast = int.from_bytes(real_sample, byteorder='big',signed=True)
    imag_sample = output_file.read(2)
    imag_sample_cast = int.from_bytes(imag_sample, byteorder='big',signed=True)


    real_sample_binary = bin(abs(real_sample_cast))[2:].zfill(16)
    imag_sample_binary = bin(abs(imag_sample_cast))[2:].zfill(16)

    print(real_sample_binary + " " + imag_sample_binary)
    counter +=1