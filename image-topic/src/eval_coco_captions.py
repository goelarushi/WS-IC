import os
import os.path as osp
import numpy as np
import json
import sys

this_dir = osp.dirname(osp.realpath((__file__)))
sys.path.insert(0, osp.join(this_dir,'coco-caption-master'))

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
def bleu_eval(resFile):
    data_dir = osp.join(this_dir,'..','..', 'coco/annotations/2014')
    result_dir = osp.join(this_dir,'..','fake_captions')
    dataType='test2014'
    algName = 'fakecap'
    # annFile='%s/captions_%s.json'%(data_dir,dataType)
    annFile = '%s/%s.json' % (data_dir, dataType)
    subtypes=['results', 'evalImgs', 'eval']
    # [resFile, evalImgsFile, evalFile]= \
    # ['%s/random_captions_%s_%s_%s.json'%(result_dir,dataType,algName,subtype) for subtype in subtypes]

    # create coco object and cocoRes object
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    # print output evaluation scores
    # for metric, score in cocoEval.eval.items():
    #     print '%s: %.3f'%(metric, score)
