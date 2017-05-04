import os,path,sys,shutil
import numpy as np
import argparse
from vae import BatchGenerator,VAE

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nBatch","-b",dest="nBatch",type=int,default=64)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=2e-3)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="models")
    parser.add_argument("--reload","-l",dest="reload",type=str,default=None)
    args = parser.parse_args()
    args.zdim = 20

    batch = BatchGenerator()
    vae = VAE(isTraining=True,imageSize=[28,28],labelSize=10,args=args)

    vae.train(f_batch=batch.getBatch)
