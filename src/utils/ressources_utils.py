import psutil

def info_cpu():
    print('### CPU (count - all)         : {}'.format(psutil.cpu_count()))
    print('### CPU (count - physical)    : {}'.format(psutil.cpu_count(logical=False)))
    print('### CPU (count - reference nb): {}'.format(len(psutil.Process().cpu_affinity())))
    print(' ')
    
def info_details_mem(text=''):
    mem = psutil.virtual_memory()
    print(text)
    #print(mem)
    conv=1024**3
    print('### Memory total     {:.2f} Gb'.format(mem.total/conv))
    print('### Memory percent   {:.2f} %'.format(mem.percent))
    print('### Memory available {:.2f} Gb'.format(mem.available/conv))
    print('### Memory used      {:.2f} Gb'.format(mem.used/conv))
    print('### Memory free      {:.2f} Gb'.format(mem.free/conv))
    print('### Memory active    {:.2f} Gb'.format(mem.active/conv))
    print('### Memory inactive  {:.2f} Gb'.format(mem.inactive/conv))
    print('### Memory buffers   {:.2f} Gb'.format(mem.buffers/conv))      
    print('### Memory cached    {:.2f} Gb'.format(mem.cached/conv))    
    print('### Memory shared    {:.2f} Gb'.format(mem.shared/conv))   
    print('### Memory slab      {:.2f} Gb'.format(mem.slab/conv)) 
    print(' ')
    
def info_mem(text=''):
    mem = psutil.virtual_memory()
    print(text)
    conv=1024**3
    print('### Memory total/available/used  {:.2f} Gb/{:.2f} Gb/{:.2f} Gb'.format(mem.total/conv,mem.available/conv,mem.used/conv))
    print('### Memory used   {:.2f} % {:.2f} Gb'.format(mem.percent,mem.used/conv))
    print(' ')
    
def mem_df(df, text=''):
    print(text)
    mem=sum(df.memory_usage().to_dict().values())
    print('### Memory dataframe {:.2f} Gb/  {:.2f} Mb/'.format(mem/1024**3,mem/1024**2))
          
def mem_details_df(df, text=''):
    print(text)
    conv=1024**3
    print('### Memory dataframe:')
    print(df.info(memory_usage="deep"))
        