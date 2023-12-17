import numpy as np 
from typing import List
import concurrent.futures

class MainTool: 

    @staticmethod
    def send_task_with_futures(task:object, args: List[tuple], 
                               max_workers=4) -> List: 
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
        futures = [executor.submit(task, a) for a in args]
        concurrent.futures.wait(futures)
        return [f.result() for f in futures]

    @staticmethod
    def dictlist_to_listdict(mydict: dict): 
        return [dict(zip(mydict,t)) for t in zip(*mydict.values())]
    
    @staticmethod
    def listdict_to_dictlist_samekey(mylist: List[dict]): 
        return {k: [dic[k] for dic in mylist] for k in mylist[0]}
    
    @staticmethod
    def listdict_to_dictlist_diffkey(mylist: List[dict]): 
        common_keys = set.intersection(*map(set, mylist))
        return {k: [dic[k] for dic in mylist] for k in common_keys}
    
    @staticmethod
    def listdict_to_dictlist(mylist: List[dict]): 
        out = {}
        [out.update(i) for i in mylist]
        return out 

    @staticmethod
    def convert_to_numpy_array(x:float or List[float]): 
        if isinstance(x, list): return np.array(x)
        else: return x
    
    @staticmethod
    def convert_array_to_list(x: float or np.array): 
        if isinstance(x, float): return [x]
        else: return list(x)
    
    @staticmethod
    def convert_any_to_list(x): 
        if not isinstance(x, list): return [x]
        else: return x
    
    @staticmethod
    def flatten_list(x:List[List]) -> List: 
        return [item for sublist in x for item in sublist]

    @staticmethod
    def order_join_lists(keys:List, values:List, 
                        by_key:bool = True, reverse = False) -> dict: 
        data = dict(zip(keys,values))
        if by_key: return dict(sorted(data.items(), reverse=reverse))
        else: return sorted(data.items(), key=lambda x: x[1], reverse=reverse)   



    





