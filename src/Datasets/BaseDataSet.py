import torch
import torch.utils.data
class StreamDataSetHelper():
    '''first dimsion of data is seq length'''
    def __init__(self,example_data_list,window_size = 1):
        self.length = 0
        self.prefix_list = [0]
        for i in range(len(example_data_list)):
            self.length+=example_data_list[i].shape[0]-window_size+1
            self.prefix_list.append(self.length)
    def __binary_search(self,idx,start,end):
        middle = (start+end)//2
        if(self.prefix_list[middle]<=idx and self.prefix_list[middle+1]>idx):
            return middle
        elif(self.prefix_list[middle+1]<=idx):
            return self.__binary_search(idx,middle+1,end)
        elif(self.prefix_list[middle]>idx):
            return self.__binary_search(idx,start,middle)
        else:
            assert False

    def __len__(self):
        return self.length
    def __getitem__(self, item):

        results = self.__binary_search(item,0,len(self.prefix_list)-1)
        assert results>=0
        return results,item-self.prefix_list[results]

