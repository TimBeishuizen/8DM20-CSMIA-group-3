# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 09:30:15 2017

@author: pmoeskops
"""

import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import theano
import lasagne
import time
import random
random.seed(0)
import glob
import PIL.Image
import os

def buildLeNet(X1):
    inputlayer = lasagne.layers.InputLayer(shape=(None, 1, 32, 32),input_var=X1)    
    print inputlayer.output_shape
    
    layer1 = lasagne.layers.Conv2DLayer(inputlayer, num_filters=6, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    print layer1.output_shape 
    
    layer2 = lasagne.layers.MaxPool2DLayer(layer1, pool_size=(2, 2))
    print layer2.output_shape 
    
    layer3 = lasagne.layers.Conv2DLayer(layer2, num_filters=16, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    print layer3.output_shape 
    
    layer4 = lasagne.layers.MaxPool2DLayer(layer3, pool_size=(2, 2))
    print layer4.output_shape 
    
    layer4 = lasagne.layers.flatten(layer4)
    print layer4.output_shape 
    
    layer5 = lasagne.layers.DenseLayer(layer4,num_units=120,nonlinearity=lasagne.nonlinearities.rectify)    
    print layer5.output_shape 
    
    layer6 = lasagne.layers.DenseLayer(layer5,num_units=84,nonlinearity=lasagne.nonlinearities.rectify)
    print layer6.output_shape 
    
    outputlayer = lasagne.layers.DenseLayer(layer6,num_units=2,nonlinearity=lasagne.nonlinearities.softmax)     
    print outputlayer.output_shape 
    
    return layer1, layer2, layer3, layer4, layer5, layer6, outputlayer
    
def make2Dpatches(samples, batch, images, patchsize, label):
    
    halfsize = int(patchsize/2)
    
    X = np.empty([len(batch),1,patchsize,patchsize],dtype=np.float32)
        
    Y = np.zeros((len(batch),2),dtype=np.int16) 
        
    for i in xrange(len(batch)):
        
        patch = images[samples[0][batch[i]],(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize),(samples[2][batch[i]]-halfsize):(samples[2][batch[i]]+halfsize)]
       
        X[i,0] = patch        
        Y[i,label] = 1 
        
    return X, Y
    
def make2Dpatchestest(samples, batch, image, patchsize):
    
    halfsize = int(patchsize/2)
    
    X = np.empty([len(batch),1,patchsize,patchsize],dtype=np.float32)
             
    for i in xrange(len(batch)):
        
        patch = image[(samples[0][batch[i]]-halfsize):(samples[0][batch[i]]+halfsize),(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize)]
       
        X[i,0] = patch  
        
    return X
    
def loadImages(impaths,maskpaths,segpaths):
    
    images = []
    masks = []
    segmentations = []    
    
    for i in xrange(len(impaths)):
        image = np.array(PIL.Image.open(impaths[i]),dtype=np.int16)[:,:,1]
        mask = np.array(PIL.Image.open(maskpaths[i]),dtype=np.int16)
        segmentation = np.array(PIL.Image.open(segpaths[i]),dtype=np.int16)
        
        images.append(image)
        masks.append(mask)
        segmentations.append(segmentation)   
        
    images = np.array(images)
    masks = np.array(masks)
    segmentations = np.array(segmentations)    
    
    return images, masks, segmentations
    
def save_weights(filename,network):
    with open(filename, 'wb') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(network), f)
        
def load_weights(filename, network):
    with open(filename, 'rb') as f:
        lasagne.layers.set_all_param_values(network, cPickle.load(f))
    
def main():
    
    project_path = os.getcwd()
    impaths = glob.glob(project_path + r'\8DM20_image_dataset\training\images\*.tif')
    maskpaths = glob.glob(project_path + r'\8DM20_image_dataset\training\1st_manual\*.gif')
    segpaths = glob.glob(project_path + r'\8DM20_image_dataset\training\mask\*.gif')
    
    print len(impaths)
    print len(maskpaths)
    print len(segpaths)
    
    if (len(impaths)!=len(segpaths) or len(impaths)!=len(maskpaths)):    
        print 'Number of images not equal'
        return
    
    images, masks, segmentations = loadImages(impaths,maskpaths,segpaths)
    
    print images.shape
    print masks.shape
    print segmentations.shape
    
    patchsize = 32
    halfsize = patchsize/2
    
    images = np.pad(images,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    masks = np.pad(masks,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    segmentations = np.pad(segmentations,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    
    print images.shape
    print masks.shape
    print segmentations.shape

        
    positivesamples = np.nonzero(segmentations*masks)
    negativesamples = np.nonzero(masks-segmentations)   
   
    print len(positivesamples[0])
    print len(negativesamples[0])
   

    minibatchsize = 200
    minibatches = 5000
    
    trainnetwork = True
    
    X = theano.tensor.tensor4()
    Y = theano.tensor.matrix()
    layer1, layer2, layer3, layer4, layer5, layer6, outputlayer = buildLeNet(X)
    outputtrain = lasagne.layers.get_output(outputlayer) 
    trainloss = lasagne.objectives.categorical_crossentropy(outputtrain, Y).mean() 
    params = lasagne.layers.get_all_params(outputlayer, trainable=True) 
    updates = lasagne.updates.adam(trainloss, params, learning_rate=0.001) 
    train = theano.function(inputs=[X, Y], outputs=trainloss, updates=updates, allow_input_downcast=True) 
    
    outputtest = lasagne.layers.get_output(outputlayer, deterministic=True) 
    test = theano.function(inputs=[X], outputs=outputtest, allow_input_downcast=True) 
   
    if trainnetwork: 
        losslist = []
        
        for i in xrange(minibatches):
            
            posbatch = random.sample(range(len(positivesamples[0])),minibatchsize/2)
            negbatch = random.sample(range(len(negativesamples[0])),minibatchsize/2)
             
            Xpos, Ypos = make2Dpatches(positivesamples,posbatch,images,32,1)
            Xneg, Yneg = make2Dpatches(negativesamples,negbatch,images,32,0)
          
            Xtrain = np.vstack((Xpos,Xneg))
            Ytrain = np.vstack((Ypos,Yneg))
           
            loss = train(Xtrain,Ytrain)
            losslist.append(loss)
            print 'Batch: {}'.format(i)
            print 'Loss: {}'.format(loss)
                
        
        plt.close('all')
        plt.figure()
        plt.plot(losslist)    
        
        save_weights(project_path + r'\8DM20_image_dataset\trainednetwork.pkl',outputlayer)
    
    else:
        load_weights(project_path + r'D:\8DM20_image_dataset\trainednetwork.pkl',outputlayer)
    
    #test the trained network
    
    testimage = np.array(PIL.Image.open(project_path + r'\8DM20_image_dataset\test\images\01_test.tif'),dtype=np.int16)[:,:,1]
    testmask = np.array(PIL.Image.open(project_path + r'\8DM20_image_dataset\test\mask\01_test_mask.gif'),dtype=np.int16)
    
    testimage = np.pad(testimage,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    testmask = np.pad(testmask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    
    testsamples = np.nonzero(testmask)
    
    print len(testsamples)
    print len(testsamples[0])    
    
    probimage = np.zeros(testimage.shape)
    
    probabilities = np.empty((0,))
    
    for i in xrange(0,len(testsamples[0]),minibatchsize):
        print i
        if i+minibatchsize < len(testsamples[0]):
            testbatch = np.arange(i,i+minibatchsize)        
        else:
            testbatch = np.arange(i,len(testsamples[0]))        
        
        Xtest = make2Dpatchestest(testsamples,testbatch,testimage,patchsize)
                
        prob = test(Xtest)
        probabilities = np.concatenate((probabilities,prob[:,1]))     
      

    for i in xrange(len(testsamples[0])):
        probimage[testsamples[0][i],testsamples[1][i]] = probabilities[i]

        
    plt.figure()
    plt.imshow(probimage,cmap='Greys_r')
    plt.axis('off')

    return   
     
    
if __name__=="__main__":
    main()