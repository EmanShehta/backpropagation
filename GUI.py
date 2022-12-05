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

def Run():
    win = Tk()

    '''
        Define Variables 
    '''
    Learning_rate = DoubleVar()
    Layers = IntVar()
    Function = StringVar(win)
    neuron = StringVar()
    Epochs = IntVar()
    BiasVAriable = IntVar()

    def Execute():
        neurons = neuron.get().split(',')
        neurons = list(map(str.strip, neurons))
        try:
            neurons = [eval(i) for i in neurons]
        except:
            messagebox.showerror("Error","Please Remove last ( ',' ) or add Number after it")
        if(len(neurons) <= int(Layers.get())):
            length = len(neurons)
            if length < int(Layers.get()):
                for i in range(int(Layers.get()) - length):
                    neurons.append(neurons[length-1])
        NN = NeuralNetwork(col, Unique_Acutal_Y, Learning_rate.get(), Epochs.get(), Data, BiasVAriable.get(),
                           Layers.get(), neurons, Function.get())
        Accuracy = NN.Tarin()

    win.geometry("800x600")
    win.eval('tk::PlaceWindow . center')

    Function.set("Function name")
    menu = OptionMenu(win, Function, 'tanh','sigmoid').place(x=30, y=180)
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
    button = Button(win, text="click Me", command=Execute).pack()
    win.mainloop()
