import json
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--tdat-filename', type=str, default='/nfs/projects/funcom/data/javastmt_fc/output/tdats.test')
    parser.add_argument('--com-filename', type=str, default='/nfs/projects/funcom/data/javastmt_fc/output/coms.test')
    parser.add_argument('--out-filename', type=str, default='./dataset/funcom_q90_test.json')

    args = parser.parse_args()
    tdat_filename = args.tdat_filename
    com_filename = args.com_filename
    out_filename = args.out_filename

    newdat = list()

    q90_train_tdats = open(tdat_filename, 'r')
    q90_train_coms = open(com_filename, 'r')

    lineA = q90_train_tdats.readline()
    lineB = q90_train_coms.readline()

    i = 0

    while lineA and lineB:
        (fidA, tdat) = lineA.split('<SEP>')
        (fidB, com) = lineB.split('<SEP>')

        fidA = int(fidA)
        fidB = int(fidB)

        if fidA != fidB:
            print('error: fids do not match in data files')
            quit()

        tdat = tdat.strip()
        com = com.strip()

        samp = dict()
        samp['fid'] = fidA
        samp['code'] = f'{tdat}'
        samp['summary'] = com

        newdat.append(samp)

        i += 1
    #if(i % 10 == 0):
    #    print(i)
    #    break

        lineA = q90_train_tdats.readline()
        lineB = q90_train_coms.readline()


    q90_train_tdats.close()
    q90_train_coms.close()

    newdats = json.dumps(newdat, indent=2)
    print(i)
    with open(out_filename, 'w', encoding='utf8') as json_file:
        json_file.write(newdats)

