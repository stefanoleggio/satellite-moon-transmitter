import sys
import json

if __name__ == '__main__':
    file = open(sys.argv[1], "r")
    PRN_codes = []
    for line in file:
        code = line.split(";")
        code[1] = code[1].rstrip()
        PRN_codes.append(code)

    #Select the first code
    code_id = PRN_codes[0][0]
    
    code_sequence = bin(int(PRN_codes[0][1],16))[2:]

    code_lenght = len(code_sequence)

    print("Lenght of the code sequence: " + str(code_lenght))

    start_mask = ''

    for code_sequence_bit in code_sequence:
        if(code_sequence_bit != '1'):
            break
        start_mask += '0'

    code_sequence_inverse = start_mask + bin((1 << code_lenght) - 1 - int(code_sequence,2))[2:]

    output = {
        'code_id': code_id,
        'code_lenght': code_lenght,
        'code_sequence': code_sequence,
        'code_sequence_inverse': code_sequence_inverse
    }

    json_object = json.dumps(output, indent = 4)
    
    with open("PRNCodes.json", "w") as outfile:
        outfile.write(json_object)