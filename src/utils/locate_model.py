import re
import os
def locate_model(check_file:str,epoch):
    if(epoch=='last'):
        check_file+="last.ckpt"
        return check_file
    dirs = os.listdir(check_file)
    for dir in dirs:
        st = "epoch=" + epoch + "-step=\d+.ckpt"
        out = re.findall(st, dir)
        if (len(out) > 0):
            check_file += out[0]
            print(check_file)
            return check_file
