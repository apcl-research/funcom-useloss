import sys
import javalang
from timeit import default_timer as timer
import tensorflow.keras as keras
import tensorflow.keras.utils
import numpy as np
import tensorflow as tf
import networkx as nx
import random
import threading
import sqlite3
from collections import Counter

import tokenizer

start = 0
end = 0

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))
            
class batch_gen(tensorflow.keras.utils.Sequence):
    def __init__(self, seqdata, extradata, tt, config, training=True):
        self.comvocabsize = config['comvocabsize']
        self.currid=0
        self.idlist = list()
        self.tt = tt
        self.batch_size = config['batch_size']

        self.extradata = extradata

        self.seqdata = dict()
        self.seqdata['dt%s' % tt] = seqdata.get('/dt%s' % tt)
        self.seqdata['ds%s' % tt] = seqdata.get('/ds%s' % tt)
        self.seqdata['s%s' % tt] = seqdata.get('/s%s' % tt)
        self.seqdata['c%s' % tt] = seqdata.get('/c%s' % tt)

        self.allfidlocs = list(range(0, np.array(self.seqdata['dt%s' % tt]).shape[0]))
        self.config = config
        self.training = training
        
        if not self.training:
            comstart = np.zeros(self.config['comlen'])
            stk = self.extradata['comstok'].w2i['<s>']
            comstart[0] = stk
            self.comstart = comstart
        random.shuffle(self.allfidlocs)

    def __getitem__(self, idx):
        ogsteps = int(np.ceil(np.array(self.seqdata['dt%s' % (self.tt)]).shape[0]/self.batch_size))
        idx = self.currid
        currfids = int(idx/self.config['comlen'])
        currword = int(idx%self.config['comlen'])
        start = (currfids*self.batch_size)
        end = self.batch_size*(currfids+1)
        batchfidlocs = self.allfidlocs[start:end]
        self.currid=(self.currid+1)%ogsteps
        return self.qs_make_batch(batchfidlocs, currword)

    def qs_make_batch(self, batchfidlocs, currword):
        return self.qs_divideseqs(batchfidlocs, currword)

    def __len__(self):
        return int(np.ceil(np.array(self.seqdata['dt%s' % (self.tt)]).shape[0]/self.batch_size))*int(self.config['comlen'])

    def on_epoch_end(self):
        self.currid = 0
        random.shuffle(self.allfidlocs)

    def qs_divideseqs(self, batchfidlocs, currword):
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlseqs = list()
        smlpaths = list()
        smlnodes = list()
        smledges = list()
        callnodes = list()
        calledges = list()
        bios = list()
        comouts = list()
        badfids = list()

        fiddat = dict()
        
        with_tdats = False
        with_sdats = False
        with_coms = False
        with_smlseqs = False
        with_smlpaths = False
        with_smlnodes = False
        with_smledges = False
        with_callnodes = False
        with_calledges = False
        with_bio = False

        comseqpos = 0
        
        for c, inp in enumerate(self.config['batch_config'][0]):
            if inp == 'tdat':
                with_tdats = True
            elif inp == 'sdat':
                with_sdats = True
            elif inp == 'com':
                with_coms = True
                comseqpos = c
            elif inp == 'smlseq':
                with_smlseqs = True
            elif inp == 'smlpath':
                with_smlpaths = True
            elif inp == 'smlnode':
                with_smlnodes = True
            elif inp == 'smledge':
                with_smledges = True
            elif inp == 'callnode':
                with_callnodes = True
            elif inp == 'calledge':
                with_calledges = True
            elif inp == 'bio':
                with_bio = True

        for fidloc in batchfidlocs:

            fid = self.config['locfid']['c%s' % self.tt][fidloc]

            wcomseq = self.seqdata['c%s' % self.tt][fidloc]
            wcomseq = wcomseq[:self.config['comlen']]
            #fid = wcomseq[:1][0]
            #wcomseq = wcomseq[1:self.config['comlen']+1]

            if with_tdats:
                #wtdatseq = self.get_tdat(fid, self.config['tdatlen'], filedb, filedbcur)
                wtdatseq = self.seqdata['dt%s' % self.tt][fidloc]
                wtdatseq = wtdatseq[:self.config['tdatlen']]

            if with_sdats:
                wsdatseq = self.seqdata['ds%s' % self.tt][fidloc]
                wsdatseq = wsdatseq[:self.config['sdatlen']]
                wsdatseq = np.delete(wsdatseq, slice(self.config['stdatlen'],None), 1)

            if with_smlseqs:
                wsmlseq = self.seqdata['s%s' % self.tt][fidloc]
                wsmlseq = wsmlseq[:self.config['smllen']]

            if with_smlnodes or with_smlpaths:
                wsmlnodes = self.extradata['s%s_nodes' % (self.tt)][fid]
                wsmlnodeslen = len(wsmlnodes)

                # crop/expand ast sequence
                wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
                tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
                tmp[:wsmlnodes.shape[0]] = wsmlnodes
                wsmlnodes = np.int32(tmp)

            if with_smledges or with_smlpaths:
                wsmledges = self.extradata['s%s_edges' % (self.tt)][fid]
                if (wsmledges.shape[0] > 1000):
                    badfids.append(fid)
                    continue
            
                # crop/expand ast adjacency matrix to dense
                wsmledges = np.asarray(wsmledges.todense())
                wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
                tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
                tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
                wsmledges = np.int32(tmp)

            if with_smlpaths:
                g = nx.from_numpy_matrix(wsmledges)
                astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
                wsmlpaths = list()

                for astpath in astpaths:
                    source = astpath[0]
                    
                    if len([n for n in g.neighbors(source)]) > 1:
                        continue
                    
                    for path in astpath[1].values():
                        if len([n for n in g.neighbors(path[-1])]) > 1:
                            continue # ensure only terminals as in Alon et al
                        
                        if len(path) > 1 and len(path) <= self.config['pathlen']:
                            newpath = self.idx2tok(wsmlnodes, path)
                            tmp = [0] * (self.config['pathlen'] - len(newpath))
                            newpath.extend(tmp)
                            wsmlpaths.append(newpath)
                
                random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
                wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
                if len(wsmlpaths) < self.config['maxpaths']:
                    wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
                wsmlpaths = np.asarray(wsmlpaths)

            if with_callnodes:
                wcallnodes = np.asarray(self.extradata['callnodedata'][fid])

                # cropping call chain nodes
                wcallnodes = wcallnodes[:self.config['maxcallnodes'],:self.config['tdatlen']]
                tmp3 = np.zeros((self.config['maxcallnodes'],self.config['tdatlen']), dtype='int32')
                tmp3[:wcallnodes.shape[0],:wcallnodes.shape[1]] = wcallnodes
                wcallnodes = np.int32(tmp3)


            if with_calledges:
                wcalledges = self.extradata['calledges'][fid]
  
                # cropping call chain edges
                wcalledges = np.asarray(wcalledges.todense())
                wcalledges = wcalledges[:self.config['maxcallnodes'], :self.config['maxcallnodes']]
                tmp2 = np.zeros((self.config['maxcallnodes'], self.config['maxcallnodes']), dtype='int32')
                tmp2[:wcalledges.shape[0], :wcalledges.shape[1]] = wcalledges
                wcalledges = np.int32(tmp2)
                
            if with_bio:
                humanattnres = self.extradata['biodats'][fid]
                
                humanattnres = humanattnres[:wsmlnodeslen]
                
                whumanattn = list()
                for s in humanattnres:
                    whumanattn.append(s[0])
                
                for i in range(self.config['maxastnodes'] - wsmlnodeslen):
                    whumanattn.append(1)
                
                whumanattn = np.asarray(whumanattn, dtype="float32")

            if not self.training:
                wcomseq = self.comstart
                inps = list()
                for inp in self.config['batch_config'][0]:
                    if inp == 'tdat':
                        inps.append(wtdatseq)
                    elif inp == 'sdat':
                        inps.append(wsdatseq)
                    elif inp == 'com':
                        inps.append(wcomseq)
                    elif inp == 'smlseq':
                        inps.append(wsmlseq)
                    elif inp == 'smlpath':
                        inps.append(wsmlpaths)
                    elif inp == 'smlnode':
                        inps.append(wsmlnodes)
                    elif inp == 'smledge':
                        inps.append(wsmledges)
                    elif inp == 'callnode':
                        inps.append(wcallnodes)
                    elif inp == 'calledge':
                        inps.append(wcalledges)
                    elif inp == 'bio':
                        inps.append(whumanattn)
                comseqs.append(wcomseq)
                fiddat[fid] = inps
            else:
                if with_tdats:
                    tdatseqs.append(wtdatseq)
                if with_sdats:
                    sdatseqs.append(wsdatseq)
                if with_smlseqs:
                    smlseqs.append(wsmlseq)
                if with_smlpaths:
                    smlpaths.append(wsmlpaths)
                if with_smlnodes:
                    smlnodes.append(wsmlnodes)
                if with_smledges:
                    smledges.append(wsmledges)
                if with_callnodes:
                    callnodes.append(wcallnodes)
                if with_calledges:
                    calledges.append(wcalledges)
                if with_bio:
                    bios.append(whumanattn)
                comseq = wcomseq[0:currword]
                comout = wcomseq[currword]
                comout = keras.utils.to_categorical(comout, num_classes=self.comvocabsize)
                for j in range(0, len(wcomseq)):
                    try:
                        comseq[j]
                    except IndexError as ex:
                        comseq = np.append(comseq, 0)
                comseqs.append(comseq)
                comouts.append(np.asarray(comout))

        if self.training:
            if with_tdats:
                tdatseqs = np.asarray(tdatseqs)
            if with_sdats:
                sdatseqs = np.asarray(sdatseqs)
            if with_smlseqs:
                smlseqs = np.asarray(smlseqs)
            if with_smlpaths:
                smlpaths = np.asarray(smlpaths)
            if with_smlnodes:
                smlnodes = np.asarray(smlnodes)
            if with_smledges:
                smledges = np.asarray(smledges)
            if with_callnodes:
                callnodes = np.asarray(callnodes)
            if with_calledges:
                calledges = np.asarray(calledges)
            if with_bio:
                bios = np.asarray(bios)
                
            comseqs = np.asarray(comseqs)
            comouts = np.asarray(comouts)
            
            inps = list()
            oups = list()
            
            for inp in self.config['batch_config'][0]:
                if inp == 'tdat':
                    inps.append(tdatseqs)
                elif inp == 'sdat':
                    inps.append(sdatseqs)
                elif inp == 'com':
                    inps.append(comseqs)
                elif inp == 'smlseq':
                    inps.append(smlseqs)
                elif inp == 'smlpath':
                    inps.append(smlpaths)
                elif inp == 'smlnode':
                    inps.append(smlnodes)
                elif inp == 'smledge':
                    inps.append(smledges)
                elif inp == 'callnode':
                    inps.append(callnodes)
                elif inp == 'calledge':
                    inps.append(calledges)
                elif inp == 'bio':
                    inps.append(bios)
        
            for oup in self.config['batch_config'][1]:
                if oup == 'comout':
                    oups.append(comouts)
                
            if len(oups) == 1:
                oups = oups[0]
        
        if not self.training:
            return (fiddat, badfids, comseqpos)
        else:
            # print()
            # print(len(inps[1][0]))
            # sys.exit()
            # print('f0:')
            # print(inps[0][0])
            # print(inps[1][0])
            # print(np.argmax(oups[0]))
            # print('f1:')
            # print(inps[0][1])
            # print(inps[1][1])
            # print(np.argmax(oups[1]))
            # print('f5:')
            # print(inps[0][5])
            # print(inps[1][5])
            # print(np.argmax(oups[5]))
            # print('f12:')
            # print(inps[0][12])
            # print(inps[1][12])
            # print(np.argmax(oups[12]))
            # print('f13:')
            # print(inps[0][13])
            # print(inps[1][13])
            # print(np.argmax(oups[13]))
            # print('f18:')
            # print(inps[0][18])
            # print(inps[1][18])
            # print(np.argmax(oups[18]))
            # print()
            # print('oups:', type(oups), '\n', np.array(oups).shape)
            # for i in oups[:26]:
            #     print(np.argmax(i), end=' ')
            #     print()
            # print()
            # sys.exit()
            return (inps, oups)

    def idx2tok(self, nodelist, path):
        out = list()
        for idx in path:
            out.append(nodelist[idx])
        return out
