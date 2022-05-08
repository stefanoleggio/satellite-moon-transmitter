if __name__ == '__main__':

    chunk_len = 10 #Size of bit for chunk

    message = open("message.bin", "rb")

    PRN_SEQUENCE = 0b1000101010010 #Example of PRN sequence

    iter_count = 0
    eof = False

    while(not eof):

        chunk = []

        while(iter_count < chunk_len):

            message_bit = message.read(1)

            iter_count += 1

            # Splitting the message in chunks
            try:
                chunk.append(int(message_bit))
            except:
                eof=True
                break


        if(len(chunk) < 1):
            break
        
        print("Message chunk")
        print(chunk)

        # Adding PRN

        chunk_PRN = []
        for message_bit in chunk:
            if(message_bit):
                chunk_PRN.append(PRN_SEQUENCE)
            else:
                chunk_PRN.append(~PRN_SEQUENCE)

        print("Chunk with PRN")
        print(chunk_PRN)

        print("\n")
        
        iter_count = 0