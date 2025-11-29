import matplotlib.pyplot as plt
import numpy as np
def Show_true_predict_points(test_mae, predictions_list, tif_name=None):
    true=[]
    prec=[]
    for set in predictions_list:
        if set[0][0]==0 or set[1][0]==0:
            continue
        true.append(set[1][0])
        prec.append(set[0][0])
    pmin,pmax=min(true+prec),max(true+prec)
    y=x=np.linspace(pmin,pmax,1000)
    plt.figure()
    plt.plot(x,y,color="red")
    plt.scatter(true,prec,color="blue")
    plt.xlabel("True")
    plt.ylabel("Predict")
    plt.title(f"test MAE={test_mae}")
    if tif_name!=None:
        plt.savefig(tif_name)
    plt.show()
def Show_loss(losses, tif_name=None):
    epoch = range(1,len(losses)+1)
    plt.figure()
    plt.plot(epoch,losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss-epoch curve")
    if tif_name!=None:
        plt.savefig(tif_name)
    plt.show()