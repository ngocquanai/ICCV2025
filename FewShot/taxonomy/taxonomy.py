import torch



def name2index(taxonomy, reversed_taxonomy) :
    # receive input is an dict
    # recursively replace inplace
    for element in taxonomy.keys() :
        if taxonomy[element] == [] :
            # Empty list
            continue
        if type(taxonomy[element]) == type([]) : # type(list)
            current_list = taxonomy[element]
            for idx in range(len(current_list)) :
                name = current_list[idx]
                current_list[idx] = reversed_taxonomy[name]
        elif type(taxonomy[element]) == type(dict()) : # a dict, it means taxonomy[element] is NOT a leaf node
            name2index(taxonomy[element], reversed_taxonomy)
        else :
            print(taxonomy[element])


def get_name_value(dictionary) :
    keys = []
    values = []
    for key, value in dictionary.items() :
        keys.append(key)
        values.append(value)
    return keys, values

def get_classname_by_layer(taxonomy) :
    layer1_name, layer1_value = get_name_value(taxonomy)

    layer2_name = []
    layer2_value = []
    for l1_value in layer1_value :
        curr_name, curr_value = get_name_value(l1_value)
        layer2_name = layer2_name + curr_name
        layer2_value = layer2_value + curr_value

    layer3_name = []
    layer3_value = []
    for l2_value in layer2_value :
        curr_name, curr_value = get_name_value(l2_value)
        layer3_name = layer3_name + curr_name
        layer3_value = layer3_value + curr_value

    return layer1_name, layer2_name, layer3_name

def flatten_taxonomy(taxonomy, result=None):
    if result is None:
        result = {}
    for key, value in taxonomy.items():
        if isinstance(value, dict):
            # Recursively flatten the dictionary
            flatten_taxonomy(value, result)
        else:
            # Add key-value pairs for leaf nodes
            result[key] = value
    return result






def get_layers_weight(class_name, taxonomy) :

    total_class = len(class_name)
    reversed_class_name = dict()

    for key, value in class_name.items() :
        if value in reversed_class_name :
            print("ERROR")
        else :
            reversed_class_name[value] = key

    name2index(taxonomy, reversed_class_name)





    layer1_name, layer2_name, layer3_name = get_classname_by_layer(taxonomy)    
    flattened_taxonomy = flatten_taxonomy(taxonomy)




    if total_class == 100 :
    ####### CIFAR100 #####
        layer3_weight = torch.zeros((24, 100))
        layer2_weight = torch.zeros((6, 24))
        layer1_weight = torch.zeros((3, 6))

        layer2_num_of_classes = [4, 7, 2, 3, 5, 3]
        layer1_num_of_classes = [2, 3, 1]
    


    elif total_class == 37 :

    ####### PETS #####
        layer3_weight = torch.zeros((12, 37))
        layer2_weight = torch.zeros((4, 12))
        layer1_weight = torch.zeros((2, 4))

        layer2_num_of_classes = [4, 2, 4, 2]
        layer1_num_of_classes = [1, 3]


    elif total_class == 101 :

    ####### FOOD101 #####
        layer3_weight = torch.zeros((25, 101))
        layer2_weight = torch.zeros((7, 25))
        layer1_weight = torch.zeros((7, 7))

        layer2_num_of_classes = [4, 6, 3, 3, 4, 3, 2]
        layer1_num_of_classes = [1, 1, 1, 1, 1, 1, 1]

    else :
        print("ERROR")


    for i in range(len(layer3_name)) :
        name = layer3_name[i]
        indices = flattened_taxonomy[name]

        for idx in indices :
            layer3_weight[i][idx] = 1



    current = 0
    for idx in range(len(layer2_num_of_classes)) :
        for i in range(current, current + layer2_num_of_classes[idx]) :
            layer2_weight[idx][i] = 1
        current += layer2_num_of_classes[idx]





    current = 0
    for idx in range(len(layer1_num_of_classes)) :
        for i in range(current, current + layer1_num_of_classes[idx]) :
            layer1_weight[idx][i] = 1
        current += layer1_num_of_classes[idx]



    return [layer1_weight, layer2_weight, layer3_weight]
