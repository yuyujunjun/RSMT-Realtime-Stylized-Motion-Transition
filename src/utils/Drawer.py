import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TKAgg')
class Drawer():
    def __init__(self):
        str_dict = {"point":"o",}
        pass
    def create_canvas(self,m,n):
        self.fig,self.ax = plt.subplots(m,n,figsize=(m,n*2))

    def get_window(self,window):
        return self.ax[window]

    def draw(self,*args,channel,feature):

        self.ax[channel,feature].plot(*args)
    def scatter(self,*args,channel,feature):
        self.ax[channel,feature].scatter(*args)
    def show(self):
        plt.show()