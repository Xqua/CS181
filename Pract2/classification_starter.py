## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.stats import entropy

import util

from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier


list_call_name = ['load_image', 'load_dll', 'open_file', 'check_for_debugger', 'get_system_directory', 'open_key', 'query_value', 'create_mutex', 'set_windows_hook', 'create_window', 'find_window', 'enum_window', 'show_window', 'open_process', 'get_file_attributes', 'create_thread', 'sleep', 'destroy_window', 'find_file', 'com_create_instance', 'vm_protect', 'enum_keys', 'enum_values', 'get_windows_directory', 'com_get_class_object', 'create_process', 'kill_process', 'thread', 'process', 'get_computer_name', 'create_key', 'read_value', 'create_directory', 'set_value', 'delete_file', 'create_open_file', 'create_file', 'copy_file', 'open_mutex', 'get_system_time', 'recv_socket', 'dump_line', 'trimmed_bytes', 'get_username', 'create_socket', 'bind_socket', 'connect_socket', 'send_socket', 'open_scmanager', 'open_service', 'get_host_by_name', 'move_file', 'set_file_attributes', 'vm_allocate', 'vm_write', 'create_thread_remote', 'impersonate_user', 'open_url', 'delete_value', 'enum_processes', 'remove_directory', 'delete_key', 'revert_to_self', 'message', 'set_file_time', 'read_section', 'connect', 'set_thread_context', 'download_file_to_cache', 'create_service', 'start_service', 'change_service_config', 'add_netjob', 'enum_modules', 'vm_read', 'write_value', 'unload_driver', 'load_driver', 'enum_share', 'download_file', 'create_namedpipe', 'create_mailslot', 'control_service', 'create_process_as_user', 'logon_as_user', 'create_interface', 'delete_share', 'listen_socket', 'enum_types', 'enum_subtypes', 'enum_items', 'get_userinfo', 'create_process_nt', 'set_system_time', 'vm_mapviewofsection', 'com_createole_object', 'accept_socket', 'delete_service', 'get_host_by_addr', 'enum_handles', 'exit_windows', 'enum_services']
dllnames = ['kernel32.dll', 'user32.dll', 'advapi32.dll', 'gdi32.dll', 'wininet.dll', 'comctl32.dll', 'shell32.dll', 'wsock32.dll', 'oleaut32.dll', 'msvbvm50.dll', 'ole32.dll', 'shlwapi.dll', 'ws2_32.dll', 'ntdll.dll', 'urlmon.dll', 'version.dll', 'crtdll.dll', 'comdlg32.dll', 'winnm.dll', 'rpcrt4.dll', 'psapi.dll', 'msvcr100.dll', 'hal.dll', 'mpr.dll', 'netapi32.dll', 'avicap32.dll', 'rasapi32.dll', 'cygwin1.dll', 'mscoree.dll', 'imagehlp.dll']




def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    X = X.todense()
    return X, feat_dict

# syscallname = []
# direc = 'test' 

# for datafile in os.listdir(direc):
#     tree = ET.parse(os.path.join(direc,datafile))
#     NAMES = extract_names(tree)
#     for name in NAMES:
#         if name not in syscallname:
#             syscallname.append(name)


def extract_names(tree):
    tmplist = []
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if el.tag not in tmplist:
                tmplist.append(el.tag)
    return tmplist

def syscall_name_counter(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c[el.tag] += 1
    entrop = measure_entropy(c)
    c['sys_call_entropy'] = entrop
    return c

# fs = []
# direc = 'test' 

# for datafile in os.listdir(direc)[:100]:
#     tree = ET.parse(os.path.join(direc,datafile))
#     fs.append(failure_success(tree))



def measure_entropy(counter):
    entrop = 0
    all_list = []
    for k in counter.keys():
        all_list.append(float(counter[k]))
    all_list = np.array(all_list)
    all_list = all_list/all_list.sum()
    entrop = entropy(all_list)
    # for i in all_list:
    #     entrop += i * np.log(i)
    if np.isnan(entrop):
        print entrop, all_list, counter
    return entrop

def failure_success(tree):
    c = Counter()
    c['successes'] = 0
    c['faliures'] = 0
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            try: 
                s = el.get('successful')
                # print s. s=='1'
                if s == '1':
                    c['successes'] += 1
                elif s == '0':
                    c['faliures'] += 1
            except:
                pass
    return c


def string_entropy(tree):
    c = Counter()
    ctmp = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            items = el.items()
            for it in items:
                ctmp[it[1]] += 1
    entrop = measure_entropy(ctmp)
    c['string_entropy'] = entrop
    return c

def dll_type(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if el.tag == 'load_dll':
                try:
                    fn = el.get('filename').split('\\')[-1].lower()
                    if fn in dllnames:
                        c[fn] += 1
                except:
                    c['noDllname'] += 1
    entrop = measure_entropy(c)
    c['dll_call_entropy'] = entrop
    return c


def Api_call_counter(tree):
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            apicall = el.get('apifunction')
            if apicall != None:
                c[apicall] += 1
    entrop = measure_entropy(c)
    c['API_call_entropy'] = entrop
    return c



## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    second_last = None
    second_first = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if second_first:
                c["second_call-"+el.tag] = 1
                second_first = False
            if first:
                c["first_call-"+el.tag] = 1
                first = False
                second_first = True
            second_last = last_call
            last_call = el.tag  # update last call seen
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    c["2ndlast_call-"+second_last] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

## The following function does the feature extraction, learning, and prediction
def main(saved_extraction=None, type_clf='tree', nb_tree=20):
    from sklearn.ensemble import RandomForestClassifier
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions.csv"
    #YOU ADD HERE THE NEW FUNCTIONS THEY HAVE TO RETURN A COUNTER CLASS
    ffs = [first_last_system_call_feats, system_call_count_feats, syscall_name_counter, dll_type, failure_success,string_entropy, Api_call_counter]
    if saved_extraction:
        X_train,global_feat_dict,t_train,train_ids = np.load('train_extract.npy')
        X_test,_,t_ignore,test_ids = np.load('test_extract.npy')
    else:
        X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
        X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
        np.save('train_extract.npy',(X_train,global_feat_dict,t_train,train_ids))
        np.save('test_extract.npy',(X_test,_,t_ignore,test_ids))
    #CrossValidation for mScoring Purposes
    print "Number of Feature used in the analysis :", X_train.shape

    Class_weight=np.array([3.69,1.62,1.2,1.03,1.33,1.26,1.72,1.33,52.14,0.68,17.56,1.04,12.18,1.91,1.3])/100.0
    if type_clf=='tree':
        clf = RandomForestClassifier(n_estimators=nb_tree)
    elif type_clf=='Etree':
        clf = ExtraTreesClassifier(n_estimators=nb_tree)
    elif type_clf=='SVC':
        clf = svm.SVC(kernel='rbf')
    if type_clf =='tree' or type_clf=='Etree':
        weight=[]
        for i in range(0,len(t_train[:int(len(X_train)*0.75)])):
            ind=t_train[i]
            weight.append(Class_weight[ind])
        clf.fit(X_train[:int(len(X_train)*0.75)], t_train[:int(len(X_train)*0.75)], sample_weight=weight)
    else:
        clf.fit(X_train[:int(len(X_train)*0.75)], t_train[:int(len(X_train)*0.75)])
    CV_hat = clf.predict(X_train[int(len(X_train)*0.75):])
    d = (t_train[int(len(X_train)*0.75):] == CV_hat)
    print "Estimation is:",float(d.sum())/len(d)
    if type_clf=='tree':
        clf = RandomForestClassifier(n_estimators=nb_tree)
    elif type_clf=='SVC':
        clf = svm.SVC(kernel='rbf')
    elif type_clf=='Etree':
        clf = ExtraTreesClassifier(n_estimators=nb_tree)
    if type_clf =='tree' or type_clf=='Etree':
        weight=[]
        for i in range(0,len(t_train)):
            ind=t_train[i]
            weight.append(Class_weight[ind])
        clf.fit(X_train, t_train, sample_weight=weight)
    else:
        clf.fit(X_train, t_train)
    t_hat = clf.predict(X_test)
    util.write_predictions(t_hat, test_ids, outputfile)

    # return X_train, global_feat_dict,t_train,train_ids

# def main():
#     train_dir = "train"
#     test_dir = "test"
#     outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument
    
#     # TODO put the names of the feature functions you've defined above in this list
#     ffs = [first_last_system_call_feats, system_call_count_feats, syscall_name_counter, dll_type, failure_success]
    
#     # extract features
#     print "extracting training features..."
#     X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
#     print "done extracting training features"
#     print
    
#     # TODO train here, and learn your classification parameters
#     print "learning..."
#     learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
#     print "done learning"
#     print
    
#     # get rid of training data and load test data
#     del X_train
#     del t_train
#     del train_ids
#     print "extracting test features..."
#     X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
#     print "done extracting test features"
#     print
    
#     # TODO make predictions on text data and write them out
#     print "making predictions..."
#     preds = np.argmax(X_test.dot(learned_W),axis=1)
#     print "done making predictions"
#     print
    
#     print "writing predictions..."
#     util.write_predictions(preds, test_ids, outputfile)
#     print "done!"

if __name__ == "__main__":
    main()
    