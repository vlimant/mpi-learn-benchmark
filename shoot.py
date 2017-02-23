import os
import sys
import itertools
import random
import time
import argparse
import socket
parser = argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--sub_masters',type=int, default=0)
parser.add_argument('--max_round', type=int, default=-1)
parser.add_argument('--validate',default=None)
parser.add_argument('--epoch', default=None)
parser.add_argument('--batch', default=None)
parser.add_argument('--worker', default=None)
parser.add_argument('--extra',default='')
options = parser.parse_args()

test=[]
train=[]

extra= options.extra
host = socket.gethostname()
cooley = ('cooley' in host)
daint = ('daint' in host)
sm = ('imperium' in host or 'culture' in host)

if cooley:
    dataloc ='/home/vlimant/Delphes/keras_archive_dustin/blobs/'
elif daint:
    dataloc = '/scratch/snx3000/vlimant/data/blobs/'
elif sm:
    dataloc = '/data/shared/Delphes/keras_archive_dustin/blobs/'

for f in os.popen('ls %s*/*/*.h5'%dataloc).read().split('\n'):
    if len(test)<7:
        test.append(f)
    else:
        train.append(f)

print len(test),"files for testing"
print len(train),"files for training"

open('train_dustin.list','w').write('\n'.join(train))
open('test_dustin.list','w').write('\n'.join(test))

## scan all
#ws = [0,1,2,4,6,8,16,24,32,64]
#bs = [100,1000,10000]
#es = [10,20,30]

## scan the scaling
ws = [16,0,1,2,4,6,8,10,12,14,20,40,60]
bs = [100]
es = [10]

fv = [0,0.25,0.5,0.75,1]
if options.validate: fv = map(float, options.validate.split(','))
if options.epoch: es = map(int, options.epoch.split(','))
if options.batch: bs = map(int, options.batch.split(','))
if options.worker: ws = map(int, options.worker.split(','))

## scan batch size
#ws = [7]
#bs = [10,100,1000,5000,10000]
#es = [10]

tests = list(itertools.product(ws,bs,es,fv))
random.shuffle(tests)

sub_masters = options.sub_masters
cpu = options.cpu
max_round = options.max_round
rounds = 0
pwd = os.getenv('PWD')

for nw,ba,ep,v in tests:
    if max_round>0 and rounds>max_round:
        break
    rounds+=1
        
    nn = nw +1 #for the master
    n_sub_masters = 0
    if sub_masters: 
        n_sub_masters=nw/sub_masters
        nn += n_sub_masters

    if sm and nn > 20:
        print "too many nodes"
        break

    if sub_masters and not n_sub_masters: continue ## no need for this
    label='nw-%d_bs-%d_ep-%d'%(nn, ba, ep)
    if extra: label = extra+"_"+label
    #label += '_bis'
    if v == 0: label+='_noval'
    elif v <1: 
        label+='_val%s'%v
        if nn == 1:
            print "no workers and fractional validation is not relevant. not running"
            continue

    if n_sub_masters: label+='_sub%d'%(n_sub_masters)## that was probably a bad convention
    if cpu : label +='_cpu'

    script = 'dannynet_%s.sh'%label
    json_out = 'dannynet_%s_history.json'%label
    log_out = 'train_%s.log'%label

    if os.path.isfile(script):
        print "\t\t",label,"already processed"
    else:
        print "Starting",label
        command = './MPIDriver.py dannynet_arch.json train_dustin.list test_dustin.list --features-name X --labels-name Y --epochs %d --loss categorical_crossentropy --batch %d --trial-name %s --master-gpu'%( ep, ba, label)
        if v!=1: command+=' --validate-fraction-every %s'% v #command+=' --validate-every 0 '
        if nn == 1: command += ' --master-only '
        if n_sub_masters: command +=' --masters %s'% (n_sub_masters+1)
        if cpu : command += ' --max-gpus 0 ' ## disable gpu
        print command

        if cooley:
            open(script,"w").write("""#!/bin/sh
NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$((NODES * 1))
cd /home/vlimant/mpi_learn_2/
mpirun -f $COBALT_NODEFILE -n $PROCS %s 
"""%( command ))
        elif daint:
            open(script,"w").write("""#!/bin/bash -l 
#SBATCH --nodes=%s
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00                                                                                                               

export CRAY_CUDA_MPS=1                                                                                                                

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                                                                                           

cd %s

export KERAS_BACKEND=theano                                                                                                           

module load daint-gpu                                                                                                                 
module load h5py/2.6.0-CrayGNU-2016.11-Python-3.5.2-serial
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0.54                                                                  

source /users/vlimant/simple/bin/activate                                                                                             

srun -n $SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK %s                                             
"""%( nn, pwd, command ))
        elif sm:
            open(script,"w").write("""#!/bin/bash
mpirun -n %s %s
"""%( nn, command ))

        os.system('chmod 755 %s'%script)
        timeout = 480 if nn<=2 else 240
        if cooley:
            os.system( 'qsub -n %d -t %d -A CMSHPCProd %s'%( nn, timeout, script ))
        elif daint:
            os.system( 'sbatch --time %d  %s'%( timeout, script ))
        elif sm:
            os.system( './%s'% script) ## this will be interactive and do things one after the other
