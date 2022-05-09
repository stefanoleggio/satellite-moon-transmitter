import json

if __name__ == '__main__':

    chunk_len = 10 #Size of bit for chunk

    message = open("message.bin", "rb")
    PRNCodes = json.load(open("PRNCodes.json", "r"))

    PRN_SEQUENCE = int(PRNCodes['code_sequence'],2)
    PRN_SEQUENCE_INVERSE = int(PRNCodes['code_sequence_inverse'],2)

    iter_count = 0
    eof = False

    while(not eof):

        chunk = []

        while(iter_count < chunk_len):

            message_bit = message.read(1)

            iter_count += 1

            # Splitting the message in chunks
            try:
                chunk.append(int(message_bit,2))
            except:
                eof=True
                break


        if(len(chunk) < 1):
            break
        
        print("Message chunk")
        print(list(map(lambda x: bin(x)[2:], chunk)))

        # Adding PRN

        chunk_PRN = []
        for message_bit in chunk:
            if(message_bit):
                chunk_PRN.append(PRN_SEQUENCE)
            else:
                chunk_PRN.append(PRN_SEQUENCE_INVERSE)

        print("Chunk with PRN")
        print(list(map(lambda x: bin(x)[2:], chunk_PRN)))

        print("\n")
        
        iter_count = 0