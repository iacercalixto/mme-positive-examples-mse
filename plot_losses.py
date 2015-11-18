import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def project_into_interval(arr, newmin=0., newmax=1.):
    """ Project data in array `arr` into interval delimited by
        [newmin,newmax] (defaults to [0.,1.]).
        Returns the projected array.
    """
    oldmin, oldmax = min(arr), max(arr)
    arr_proj = [(newmax-newmin)/(oldmax-oldmin)*(value-oldmax)+newmax
                for value in arr]
    
    return arr_proj


def plot_losses_vs_time(train_losses, valid_losses, test_losses,
                        time_measures,
                        time_label='Time',
                        title='Train x Valid losses'):
    plt.figure()
    plt.title(title)
    #train_losses = project_into_interval(train_losses)
    #valid_losses = project_into_interval(valid_losses)
    
    plt.plot(time_measures, train_losses, '-', color='b')
    plt.plot(time_measures, valid_losses, '-', color='g')
    if not len(test_losses)==0:
        plt.plot(time_measures, test_losses,  '-', color='r')
    plt.ylabel('Loss')
    plt.xlabel(time_label)
    
    if not len(time_measures)==0 or \
            len(train_losses)==0 or \
            len(valid_losses)==0:
        plt.annotate('train',
                 xy=(time_measures[-1],train_losses[-1]), 
                 xytext=(time_measures[-1],train_losses[-1]),
                 color='b')
        plt.annotate('valid',
                 xy=(time_measures[-1],valid_losses[-1]), 
                 xytext=(time_measures[-1],valid_losses[-1]),
                 color='g')
    else:
        plt.annotate('train', xy=(0,0), xytext=(0,0), color='b')
        plt.annotate('valid', xy=(10,10), xytext=(10,10), color='g')
    
    if not len(test_losses)==0:
        plt.annotate('test',
                     xy=(time_measures[-1],test_losses[-1]), 
                     xytext=(time_measures[-1],test_losses[-1]),
                     color='r')
    
    plt.show()

#time_measures = [1,3,5,7,10,14,17,20, 32]
#losses = [1., 0.9, 0.8, 0.7, .6, .55, .5, .4, .3]
#plot_losses_vs_time(losses, time_measures, time_measures, time_label='Time')