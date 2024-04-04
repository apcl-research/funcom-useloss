import json

newdat = list()

q90_train_tdats = open('/nfs/projects/funcom/data/python/output/tdats.test', 'r')
q90_train_coms = open('/nfs/projects/funcom/data/python/output/coms.test', 'r')

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
    samp['instruction'] = 'describe the following function'
    samp['input'] = tdat
    samp['output'] = com

    newdat.append(samp)

    #i += 1
    #if(i % 10 == 0):
    #    print(i)
    #    break

    lineA = q90_train_tdats.readline()
    lineB = q90_train_coms.readline()


q90_train_tdats.close()
q90_train_coms.close()

newdats = json.dumps(newdat, indent=2)

with open('funcom_python_test.json', 'w', encoding='utf8') as json_file:
    json_file.write(newdats)

