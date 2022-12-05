from tkinter import *
from NeuralNetwork import *
from tkinter import messagebox

Data = pd.read_csv('..\\penguins.csv')
NumberOfColumns = Data.shape[1]
Columns = Data.columns[1:NumberOfColumns]
Unique_Acutal_Y = Data['species'].unique()
FeacturesList = []
ClassesList = []
col =[]
for i in range(len(Columns)): col.append(Columns[i])
'''
for i in range(len(Unique_Acutal_Y)):
    j = i+1
    while(j != len(Unique_Acutal_Y)):
        ClassesList.append(Unique_Acutal_Y[i] + " With " + Unique_Acutal_Y[j])
        j=j+1
'''


def Run():
    win = Tk()

    '''
        Define Variables 
    '''
    Learning_rate = IntVar()
    Layers = IntVar()
    Function = StringVar()
    neuron = StringVar()
    Epochs = IntVar()
    BiasVAriable = IntVar()

    def show():
        neurons = neuron.get().split(',')
        neurons = list(map(str.strip, neurons))
        neurons = [eval(i) for i in neurons]
        #  Selectedclasses = classes.get().split('With')
        #  Selectedclasses = list(map(str.strip, Selectedclasses))
        # if(len(SelectedFeatures)>0 and len(Selectedclasses)>0 and Epochs.get()>0):
        NN = NeuralNetwork(col, Unique_Acutal_Y, Learning_rate.get(), Epochs.get(), Data, BiasVAriable.get(),
                           Layers.get(), neurons, Function.get())
        Accuracy = NN.Tarin()

    #        Label(win, text="Accuracy = "+str(Accuracy)).place(x=30, y=150)

    # else:
    #     messagebox.showerror('Error','Error: You have to Entered The Madnadtory Fields')

    win.geometry("800x600")
    win.eval('tk::PlaceWindow . center')
    '''
    Features.set("Select two feactures")
    OptionMenu(win, Features, *FeacturesList).place(x=100, y=100)
    # drop.pack()

    classes.set("Select two Calsses")
    OptionMenu(win, classes, *ClassesList).place(x=150, y=150)
    # drop.pack(
    '''
    fun = ['tanh'], ['sigmoid']
    Function.set("Function name")
    OptionMenu(win, Function, *fun).place(x=30, y=180)
    Label(win, text="Layers").place(x=30, y=100)
    Label(win, text="Neurons").place(x=30, y=120)
    Label(win, text="Eta").place(x=30, y=140)
    Label(win, text="Epochs").place(x=30, y=160)
    Entry(win, textvariable=Layers).place(x=100, y=100)
    Entry(win, textvariable=neuron).place(x=100, y=120)
    Entry(win, textvariable=Learning_rate).place(x=100, y=140)
    Entry(win, textvariable=Epochs).place(x=100, y=160)
    Bias_ChickButton = Checkbutton(win, text="Bias", variable=BiasVAriable, onvalue=1, offvalue=0, height=2, width=10)
    Bias_ChickButton.pack()
    button = Button(win, text="click Me", command=show).pack()
    win.mainloop()
