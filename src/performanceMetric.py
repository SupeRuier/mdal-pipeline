#%%
'''
Read the result file froms repetitions and compute the final average performance.
This whole file is used in result processing.

This file read from the result folder with the specific range of repetition number for the specific dataset, and calculate the average performance of the current combination, and plot the corresponding graphs.
'''

#%%

class PerformanceLog:
    '''
    A class to store the performance result. 
    '''
    def __init__(self, domain_id_to_names):
        self.cost_list = []
        self.domain_id_to_names = domain_id_to_names # A dict of map from domain_int to domain_name
        self.domain_id_to_names[-1] = 'Total'

        self.domain_names = list(self.domain_id_to_names.values())

        self.performance_dict = {}
        for domain_int in self.domain_id_to_names.keys():
            self.performance_dict[domain_int] = []

    def add_performance_item(self, cost, performance_dict_item):
        '''
        Add cost and performance_dict into the current PerformanceLog
        The performance_dict_item is a dictionary of domain_int and the accuracy
        '''
        self.cost_list.append(cost)
        for domain_int in self.domain_id_to_names.keys():
            self.performance_dict[domain_int].append(performance_dict_item[domain_int])

    def get_performance_dict(self):
        '''
        Return a dict with the domain name (not int) as the key.
        (Switch the keys.)
        '''
        int_to_name_performance_dict = {}
        for key, value in self.performance_dict.items():
            int_to_name_performance_dict[self.domain_id_to_names[key]] = value
        return int_to_name_performance_dict

