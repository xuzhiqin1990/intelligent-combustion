#!/usr/bin/python3
import cantera as ct
import yaml, copy
import argparse
import os

def load_chem_yaml(chem_path):
    """
    将完全的 chem_path.yaml 文件加载为字典
    """
    file = open(chem_path, 'r', encoding="utf-8")         # 获取yaml文件数据
    chem_dict = yaml.load(file.read(), Loader=yaml.FullLoader)       # 将yaml数据转化为字典
    file.close()
    return chem_dict

def find_species_lack(target_chem, refer_chem_list):
    """
    获得 base 机理中缺少的物种
    """
    ct.suppress_thermo_warnings()
    species_list = []
    gas = ct.Solution(target_chem)
    for chem in refer_chem_list:
        gas1 = ct.Solution(chem)
        # 取出在 gas1 中有而在 gas 中没有的物种
        species_lack = list(set(gas1.species_names) - set(gas.species_names))
        species_list.extend(species_lack)
    # 去重
    species_list = list(set(species_list))
    return species_list


def extract_species(filename, specie_name):
    """
    获得filename机理中specie_name物质对应信息
    """
    ct.suppress_thermo_warnings()
    present_path = filename
    chem_new = load_chem_yaml(present_path)
    index = chem_new['phases'][0]['species'].index(specie_name)
    relevant_species_info = chem_new['species'][index]
    return relevant_species_info


def reaction2str_expression(reaction):
    """
    将反应转化为字符串表达式, 换算格式为
    ' reactants 系数 reactants 系数 _ producers 系数 producers 系数 '
    例如 H2 + O2 = H2O
    变为 ' H2 1 O2 1 _ H2O 1 '
    # 如果反应存在 duplicate flag, 则必须在最后加上 ' duplicate '
    """
    reactants = reaction.reactants
    products = reaction.products
    key_list = ''
    for key in reactants:
        key_list += f' {key} {reactants[key]}'
    key_list += ' _ '
    for key in products:
        key_list += f'{key} {products[key]} '
    return key_list


def str_expression2reaction_equation(key_list):
    """
    将字符串表达式转化为反应， 换算格式为
    ' reactants 系数 reactants 系数 _ producers 系数 producers 系数 '
    例如 ' H2 1 O2 1 _ H2O 1 '
    变为 H2 + O2 = H2O
    return:
        equation: 反应方程式
    """
    reactant_list:list = key_list.split('_')[0]; product_list = key_list.split('_')[1]
    reactant_list = reactant_list.split(' '); product_list = product_list.split(' ')
    reactant_list.pop(-1); product_list.pop(0)
    print(reactant_list, product_list)
    reactants = {
        reactant_list[2*i]: float(reactant_list[2*i+1]) for i in range(int(len(reactant_list)/2))
    }
    products = {
        product_list[2*i]: float(product_list[2*i+1]) for i in range(int(len(product_list)/2))
    }
    reaction = ct.Reaction(reactants, products, ct.ArrheniusRate(38.7, 2.7, 26))
    return reaction.equation


def compare_2_str_expression(str1, str2):
    """
    比较两个字符串表达式是否相同
    我们认为的相同的含义是，两个字符串完全相同或者两个字符串中反应物和生成物的顺序相反
    例如 ' H2 1 O2 1 _ H2O 1 ' 和 ' O2 1 H2 1 _ H2O 1 ' 是相同的
    例如 ' H2 1 O2 1 _ H2O 1 ' 和 ' H2O 1 _ H2 1 O2 1 ' 也是相同的
    """
    str1 = str1.split('_'); str2 = str2.split('_')
    reactant1 = str1[0]; product1 = str1[1]
    reactant2 = str2[0]; product2 = str2[1]
    reactant1 = reactant1.split(' '); product1 = product1.split(' ')
    reactant2 = reactant2.split(' '); product2 = product2.split(' ')
    reactant1.pop(-1); product1.pop(0)
    reactant2.pop(-1); product2.pop(0)
    reactant1.sort(); product1.sort()
    reactant2.sort(); product2.sort()
    if reactant1 == reactant2 and product1 == product2:
        return True
    elif reactant1 == product2 and product1 == reactant2:
        return True
    else:
        return False
    

def compare_expression_with_expressionlist(expression, expression_list):
    """
    比较一个字符串表达式和一个字符串表达式列表中的所有字符串表达式是否相同
    """
    for expression1 in expression_list:
        if compare_2_str_expression(expression, expression1):
            return True
    return False

#合并思路 
#先对物质进行处理, 建立总机理缺少物质的列表, 记为species_lack_list,去其他机理中查找这些species的信息, 归入species_info_list
#
#再对反应进行处理, 先将shre中的reactions读成一个字典, key为‘反应物 系数 反应物 系数_生成物 系数 生成物 系数'型的字符串, value是对应的chem_file['reactions']
#
#其他chem均按照这一方法读成字典, 所有反应合并成新字典
#
#反应是可逆反应 在不同机理中可能两个生成物反应物相反, 需要剔除字典中此类重复

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge mechanism in the rcy\'s method')
    parser.add_argument('-m', '--merge_list_dir', type=str, nargs='+', help='the mechanism dir need to be merged', default=None)
    parser.add_argument('-b', '--base_chem', type=str, help='the base mechanism', default='chem_shrestha.yaml')
    parser.add_argument('-o', '--output', type=str, help='the output mechanism', default='new_chem.yaml')
    
    merge_list = parser.parse_args().merge_list_dir[0]
    base_chem = parser.parse_args().base_chem
    # chemlist 是 merge_list 代表的文件夹下的所有机理文件
    ## 使用 os 库读取 merge_list 文件夹下所有以 yaml 结尾的文件
    ### 读取文件夹下所有文件
    
    chem_list = [os.path.join(merge_list, file) for file in os.listdir(merge_list) if file.endswith('.yaml')]
    # 如果 base_chem 在 chem_list 中，将其剔除
    if base_chem in chem_list:
        chem_list.remove(base_chem)
        print(f'{base_chem} is removed from chem_list.')

    species_lack_list = find_species_lack(base_chem, chem_list)
    ct.suppress_thermo_warnings()
    base_gas = ct.Solution(base_chem)
    base_species = base_gas.species_names
    base_reactions = ct.Reaction.list_from_file(base_chem, base_gas)
    # 清除 base_reactions 中的 note：
    for reac in base_reactions:
        if 'note' in reac.input_data:
            note = reac.input_data['note']
            if '\n' in note:
                if note[0] == ' ':
                    # 删除开头所有的空格，不论是 1 个还是多个
                    note = note.lstrip()
                # 找到换行符的所有位置
                index = [i for i in range(len(note)) if note[i] == '\n']
                # 删除换行符后的空格
                for i in index:
                    if note[i+1] == ' ':
                        note = note[:i] + note[i+1:]
            reac.update_user_data({f'note':note})
    # base_gas_dict = load_chem_yaml(base_chem)
    # base_species_info = base_gas_dict['species']

    # gas_list = [ct.Solution(chem) for chem in chem_list]

    # species 添加
    new_species_info = {}
    reaction_dict_old = {}
    new_species = []
    for specie in species_lack_list:
        for present_chem in chem_list:
            tmp_gas = ct.Solution(present_chem)
            if specie in tmp_gas.species_names:
                info = extract_species(present_chem, specie)
                if isinstance(info, dict):
                    if 'note' in info['thermo']:    
                        del info['thermo']['note']
                    if 'note' in info['transport']:
                        del info['transport']['note']
                    new_species_info.update({specie: info})
                    new_species.append(tmp_gas.species(tmp_gas.species_index(specie)))
                    break
    ## 将新的物种信息加入到 base_gas_dict 中
    new_species = base_gas.species() + new_species
    # base_gas_dict['species'].extend(list(new_species_info.values()))
    # base_gas_dict['phases'][0]['species'].extend(list(new_species_info.keys()))
    # 获取 base 机理中所有反应的字符串表达式
    for index in range(len(base_reactions)):
        key_list = reaction2str_expression(base_reactions[index])
        # 若 duplicate 反应存在，则 key_list 已经在字典中, 则将 value 转化为 list, 并将新的 value 加入到 list 中
        if key_list in reaction_dict_old:
            if isinstance(reaction_dict_old[key_list], list):
                reaction_dict_old[key_list].append(base_reactions[index])
            else:
                reaction_dict_old[key_list] = [reaction_dict_old[key_list], base_reactions[index]]
        else:
            reaction_dict_old[key_list] = base_reactions[index]

    reaction_dict_old_0 = reaction_dict_old.copy()

    for chem in chem_list:
        reaction_dict_new = {}
        print(f'start to process {chem}\'s reactions.')
        ct.suppress_thermo_warnings()
        gas = ct.Solution(chem)
        all_reactions_new = gas.reactions()
        gas_dict = load_chem_yaml(chem)
        species_info_list = gas_dict['species']
        # 获取 base 机理中所有反应的字符串表达式

        for reac in all_reactions_new:
            # 获取反应的字符串表达式
            key_list = reaction2str_expression(reac)
            if not compare_expression_with_expressionlist(key_list, list(reaction_dict_old.keys())):
                # print(f'{key_list} is not in reaction_dict_old.')
                # 若 duplicate 反应存在，则 key_list 已经在字典中, 则将 value 转化为 list, 并将新的 value 加入到 list 中
                if key_list in reaction_dict_new:
                    if isinstance(reaction_dict_new[key_list], list):
                        reaction_dict_new[key_list].append(reac)
                    else:
                        reaction_dict_new[key_list] = [reaction_dict_new[key_list], reac]
                else:
                    reaction_dict_new[key_list] = reac
            if 'note' in reac.input_data:
                note = reac.input_data['note']
                # 如果换行号下一个字符是空格，删除之
                if '\n' in note:
                    if note[0] == ' ':
                        note = note.lstrip()
                    # 找到换行符的所有位置
                    index = [i for i in range(len(note)) if note[i] == '\n']
                    # 删除换行符后的空格
                    for i in index:
                        if note[i+1] == ' ':
                            note = note[:i] + note[i+1:]
                reac.update_user_data({f'note':note})
        reaction_dict_new.update(reaction_dict_old_0)
        # reaction_dict_old_0 变成了最大的字典
        reaction_dict_old_0 = reaction_dict_new.copy()
    print(f'len of reaction_dict_old_0: {len(reaction_dict_old_0)}')
    ## 在 reaction_dict_old_0 中剔除可逆反应
    ## 剔除方法为，将所有反应的 reactants 和 products 取出，若存在两个反应的 reactants 等于 products, products 等于 reactants, 则剔除其中一个
    # dict_unrep = copy.copy(reaction_dict_old_0)
    # for key1 in reaction_dict_old_0:
    #     for key2 in dict_unrep:
    #         reaction1 = reaction_dict_old_0[key1]
    #         if isinstance(reaction1, list):
    #             reaction1 = reaction1[0]
    #         reactants1 = reaction1.reactants
    #         products1 = reaction1.products
    #         reaction2 = reaction_dict_old_0[key2]
    #         if isinstance(reaction2, list):
    #             reaction2 = reaction2[0]
    #         reactants2 = reaction2.reactants
    #         products2 = reaction2.products
    #         if products1 == reactants2 and reactants1 == products2:
    #             del dict_unrep[key2]
    #             break

    # base_gas_dict['reactions'] = list(dict_unrep.values())
    ## dict_unrep 中提取出新的 reactions
    new_reactions = list(reaction_dict_old_0.values())
    # new_reactions 作为列表，其中的元素含有单个元素或者列表，将其中的列表全部展开
    new_reactions = [item for sublist in new_reactions for item in (sublist if isinstance(sublist, list) else [sublist])]
    reaction_type = [type(reac) for reac in new_reactions]
    print(f'reaction type 的分组计数为 {dict(zip(reaction_type, [reaction_type.count(reac) for reac in reaction_type]))}')
    print(f'old mechanism have {base_gas.n_reactions} reactions and {base_gas.n_species} species, while new mechanism have {len(new_reactions)} reactions and {len(new_species)} species.')
    print(f'The added species are {species_lack_list}.')
    # 由 new_reactions 和 new_species 生成新的 yaml 文件
    new_gas = ct.Solution(thermo='IdealGas', 
                          kinetics='GasKinetics', 
                          species=new_species, reactions=new_reactions)
    new_chem_path = parser.parse_args().output
    new_gas.write_yaml(new_chem_path)