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

    prn_sequence_inverse = start_mask + bin((1 << code_lenght) - 1 - int(prn_sequence,2))[2:]

    boc_sequence = ''
    boc_sequence_inverse = ''

    for bit in prn_sequence:
        if(bit == '1'):
            boc_sequence += '+1-1'
            boc_sequence_inverse += '-1+1'
        else:
            boc_sequence += '-1+1'
            boc_sequence_inverse += '+1-1'

    output = {
        'prn_id': code_id,
        'prn_lenght': code_lenght,
        'prn_sequence': prn_sequence,
        'prn_sequence_inverse': prn_sequence_inverse,
        'boc_sequence': boc_sequence,
        'boc_sequence_inverse': boc_sequence_inverse
    }

    json_object = json.dumps(output, indent = 4)
    
    with open("PRNCodes.json", "w") as outfile:
        outfile.write(json_object)