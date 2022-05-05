if __name__ == '__main__':

    message = open("message.bin", "rb")

    PRN_SEQUENCE = 0b0100100101010

    output = []

    message_bit = message.read(1)

    while(message_bit):
        if(int(message_bit,2)):
            output.append(PRN_SEQUENCE)
        else:
            output.append(~PRN_SEQUENCE)
        message_bit = message.read(1)

    print(output)