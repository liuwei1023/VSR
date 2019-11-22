import os
from datetime import datetime

class Logger:
    def __init__(self):
        self.f = None
    def set_filepath(self, log_path):
        self.f = open(log_path,'w')

    def WriteLine(self, str):
        now = datetime.now()
        now_str =  '%02d-%02d %02d:%02d:%02d'%(now.month,now.day,now.hour,now.minute,now.second)
        print(now_str+"  "+str)
        self.f.write(now_str+"  "+str+'\n')
        self.f.flush()
    
    def ERROR(self, str):
        now = datetime.now()
        now_str =  '%02d-%02d %02d:%02d:%02d'%(now.month,now.day,now.hour,now.minute,now.second)
        self.f.write(now_str + "  "+str+'\n')
        self.f.flush()
        raise ValueError(str)
            
system_log = Logger()