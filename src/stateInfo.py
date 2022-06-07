#%%
'''
Use a stateinfo instance to record the current situation.
Which instances is labeled is recorded.
'''

#%%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import copy
from parameter_handler import param_handler

#%%
class StateInfo:
    '''
    Record the labeled and unlabeled index and index for each class in multi-domain setting. 
    And a collected list is a binary list to instruct the labeled position.
    '''

    def __init__(self, data, dataset_name, initial_labeled_num = 100, random_state = 1):
        '''
        Initialize the current state. 
        Only consider the situation that warm start and the initial instance are fully labeled
        Labeled and unlabeled index should be in type list.
        random_state is controled by the repitition

        Parameters
        ----------
        Y_dict: a dictionary {'domain name': [domain labels]}
        '''
        self.dataset_name = dataset_name

        # The following params are only for training set.
        self.initial_labeled_num = initial_labeled_num
        self.Y_dict = copy.deepcopy(data.train_set['Y_dict'])
        self.domain_names = param_handler.domains # list with the same order in param_handler.
        self.domain_ids = list(range(len(self.domain_names)))
        self.num_domain = len(self.domain_names)
        self.num_each_domain = [len(self.Y_dict[domain]) for domain in self.domain_names] # Number of instances for each domain.
        self.total_number = sum(self.num_each_domain) # Number of instances for each domain.

        self.labeled_info, self.unlabeled_info, self.labeled_info_boolean = self.random_init(random_state)
        
        self.domain_id_to_names, self.domain_names_to_id = self._initialize_domain_id_map()

    def random_init(self, random_state):
        '''
        Initialize the current state. 
        Only consider the situation that warm start and the initial instance are fully labeled
        Labeled and unlabeled index should be in type list.

        We need to make sure the labeled instance would contain all the labels in the domains.

        Parameters
        ----------
        random_state: generalize random state, suppose to be the repetition number.

        Returns
        ----------
        labeled_info: {'domain_name':labeled_index}
        unlabeled_info: {'domain_name': unlabeled_index}
        labeled_info_boolean: {'domain_name':[0,1,0,0,...,1]}
        Grouped in a dictionary.
        '''
        np.random.seed(random_state)

        labeled_info = {}
        unlabeled_info = {}
        labeled_info_boolean ={}

        select_num_each_domain = self._get_select_num()
        
        for i in range(self.num_domain):
            total_num_current_domain = self.num_each_domain[i]

            labeled_index = np.random.choice(total_num_current_domain, size = select_num_each_domain[i], replace=False)
            labeled_info[self.domain_names[i]] = labeled_index
            unlabeled_info[self.domain_names[i]] = np.delete(np.array(range(total_num_current_domain)), labeled_index)
            
            labeled_info_boolean_domain = np.zeros(total_num_current_domain)
            for idx in labeled_info[self.domain_names[i]]:
                labeled_info_boolean_domain[idx] = 1
            labeled_info_boolean[self.domain_names[i]] = labeled_info_boolean_domain

        return labeled_info, unlabeled_info, labeled_info_boolean

    def update(self, select_list):
        '''
        The select_list is a list of tuples ('Domain', index).
        eg.[('book',23),('electronic',555),('book',236),...('electronic',6)]
        '''
        # For each instance, delete it from unlabeled list, add it to labeled list and change the positive/negative index.
        for select_instance in select_list:
            # select_instance (domain, index)
            # Remove the index from unlabeled log.
            # Can't use the following code because it moves by index
            # np.delete(self.unlabeled_info[select_instance[0]], select_instance[1])
            self.unlabeled_info[select_instance[0]] = self.unlabeled_info[select_instance[0]][self.unlabeled_info[select_instance[0]] != select_instance[1]]
            self.labeled_info[select_instance[0]] = np.append(self.labeled_info[select_instance[0]], select_instance[1])
            self.labeled_info_boolean[select_instance[0]][select_instance[1]] = 1
    
    def _get_select_num(self):
        '''
        Return a list contains the initial selected number for each domain in order.
        '''
        initial_labeled_num = self.initial_labeled_num
        num_total = sum(self.num_each_domain)
        select_num_each_domain = []

        # According to the propotion of the domains.
        for i in range(self.num_domain):
            temp = math.floor(initial_labeled_num * (self.num_each_domain[i]/num_total))
            select_num_each_domain.append(temp)

        # Because we choose floor, there still might be available instances.
        summation = sum(select_num_each_domain)
        add_on = initial_labeled_num - summation
        for i in range(int(add_on)):
            select_num_each_domain[i] += 1
        return select_num_each_domain

    def get_initial_labeled_num(self):
        '''
        Return the initial labeled number
        '''
        return self.initial_labeled_num

    def get_max_budget(self):
    
        '''
        Return the total number on each domain
        '''
        return self.total_number

    def get_domain_names(self):
        '''
        Return the name of all the domains
        Use parameter handler doomains.
        '''
        return self.domain_names

    def get_domain_ids(self):
        '''
        Return the number of all the instances
        '''
        return self.domain_ids

    def _initialize_domain_id_map(self):
        domain_id_to_names = {}
        domain_names_to_id = {}

        for i in range(len(self.domain_names)):
            domain_id_to_names[i] = self.domain_names[i]
            domain_names_to_id[self.domain_names[i]] = i
        
        return domain_id_to_names, domain_names_to_id

    def get_domain_id_to_names(self):
        '''
        Return the domain_id_to_names dict
        '''
        return self.domain_id_to_names

    def get_domain_names_to_id(self):
        '''
        Return the domain_names_to_id dict
        '''
        return self.domain_names_to_id

    def get_dataset_name(self):
        return self.dataset_name

    def get_domain_labeled_num(self):
        domain_labeled_num = {}
        for key, value in self.labeled_info.items():
            domain_labeled_num[key] = len(value)
        return domain_labeled_num

    def get_domain_labeled_boolean(self, domain):
        return self.labeled_info_boolean[domain]

    def get_domain_labeled_info_boolean(self, domain_name):
        return self.labeled_info_boolean[domain_name].tolist()

    def build_unlabeled_matrix(self, train_set, domain):
        temp_index = self.unlabeled_info[domain]
        try:
            # for TrainSetJoint
            unlabeled_matrix = train_set.X_dict[domain][temp_index,:]
        except:
            # for TrainSetSeparate
            unlabeled_matrix = train_set.all_domain_data[domain].X_dict[domain][temp_index,:]
        return unlabeled_matrix

    def build_all_matrix(self, train_set, domain):
        try:
            # for TrainSetJoint
            all_matrix = train_set.X_dict[domain]
        except:
            # for TrainSetSeparate
            all_matrix = train_set.all_domain_data[domain].X_dict[domain]
        return all_matrix