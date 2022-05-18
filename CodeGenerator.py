import sys
import json

# Usage example: python PRNGenerator.py C7_E1B.txt 0

if __name__ == '__main__':
    file = open(sys.argv[1], "r")
    SV_ID = int(sys.argv[2])
    PRN_codes = []
    for line in file:
        code = line.split(";")
        code[1] = code[1].rstrip()
        PRN_codes.append(code)

    # Select the first code
    code_id = PRN_codes[SV_ID][0]
    
    prn_sequence = bin(int(PRN_codes[SV_ID][1],16))[2:]

    code_lenght = len(prn_sequence)

    print("Lenght of the code sequence: " + str(code_lenght))

    start_mask = ''

    for prn_sequence_bit in prn_sequence:
        if(prn_sequence_bit != '1'):
            break
        start_mask += '0'

    prn_sequence_vector = []
    prn_sequence_inverse_vector = []

    for prn_bit in prn_sequence:
        prn_sequence_vector.append(int(prn_bit))
        if(prn_bit == '0'):
            prn_sequence_inverse_vector.append(1)
        else:
            prn_sequence_inverse_vector.append(0)


    boc_sequence_vector = []
    boc_sequence_inverse_vector = []


    for prn_bit in prn_sequence_vector:
        if(prn_bit == 1):
            boc_sequence_vector.append(1)
            boc_sequence_vector.append(-1)
        else:
            boc_sequence_vector.append(-1)
            boc_sequence_vector.append(1)

    for prn_bit in prn_sequence_inverse_vector:
        if(prn_bit == 1):
            boc_sequence_inverse_vector.append(1)
            boc_sequence_inverse_vector.append(-1)
        else:
            boc_sequence_inverse_vector.append(-1)
            boc_sequence_inverse_vector.append(1)

    output = {
        'prn_id': code_id,
        'prn_lenght': code_lenght,
        'prn_sequence': prn_sequence_vector,
        'prn_sequence_inverse': prn_sequence_inverse_vector,
        'boc_sequence': boc_sequence_vector,
        'boc_sequence_inverse': boc_sequence_inverse_vector
    }

    json_object = json.dumps(output)
    
    with open("PRNCodes.json", "w") as outfile:
        outfile.write(json_object)