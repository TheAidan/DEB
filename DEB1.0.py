import numpy as np

def sigmoid(x):
  y = 1/(1+np.exp(-x))
  return y
  
def dsigmoid(y):
  y = y*(1-y)
  return y
#increase by multiples of 2

def filltrainingset():
    tr_arr_x = []
    tr_arr_y = []
    x = True

    with open('D:/Machine Learning/DEB/testval.txt', 'rU') as readinfile:
        for line in readinfile:
            #load line in as string to be parsed
            linestred = line
            if(x == True):
                 tr_arr_x.append([])
                 tr_arr_x[len(tr_arr_x)-1] = linestred.split(",")
                 for enum in range(0, len(tr_arr_x[len(tr_arr_x)-1])):
                     tr_arr_x[len(tr_arr_x)-1][enum] = int(tr_arr_x[len(tr_arr_x)-1][enum])
            if(x == False):
                tr_arr_y.append([int(linestred)])
            x = not x
    return([tr_arr_x, tr_arr_y])
    
def persistenceout(wih, who):
    np.save('D:/Machine Learning/DEB/wih_cache', wih)
    np.save('D:/Machine Learning/DEB/who_cache', who)
def persistencein(wih, who):
    wih = np.load('D:/Machine Learning/DEB/wih_cache.npy')   
    who = np.load('D:/Machine Learning/DEB/who_cache.npy')
    return(wih, who)     
            
            
            
            
                
tr_arr_x = np.array(filltrainingset()[0])
tr_arr_y = np.array(filltrainingset()[1])
inputlayerthreadin = tr_arr_x
outputlayerthreadin = tr_arr_y

#inputlayerthreadin = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,1,1,0,0,0],[0,0,0,0,0,1],[0,1,0,0,0,1],[1,1,1,1,1,1]])
#outputlayerthreadin = np.array([[0],[0],[0],[1],[1],[1]])

inputlayerpredict = np.array([0,0,0,1,0,0])
prediction = True
trainingtime = True
np.random.seed(1)



#weights
ih = 2*np.random.random((len(inputlayerthreadin[0]), len(inputlayerthreadin))) - 1
ho = 2*np.random.random((len(inputlayerthreadin), 1)) - 1
ih, ho = persistencein(ih, ho)
if(trainingtime == True):
    for enum in range(0,60000):
        
        ilayer = inputlayerthreadin
        hlayer = sigmoid(np.dot(ilayer, ih))
        olayer = sigmoid(np.dot(hlayer, ho))
  
        oerror = outputlayerthreadin - olayer
        odelta = oerror*dsigmoid(olayer)
  
        herror = odelta.dot(ho.T)
        hdelta = herror * dsigmoid(hlayer)
  
        ho += hlayer.T.dot(odelta)
        ih += ilayer.T.dot(hdelta)
    persistenceout(ih, ho)    
   
            
  



if(prediction == True):
  ilayer = inputlayerpredict
  hlayer = sigmoid(np.dot(ilayer, ih))
  olayer = sigmoid(np.dot(hlayer, ho))
  print(olayer)
  



