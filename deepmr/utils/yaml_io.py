import yaml

def read_yaml_data(yaml_file, key_name = None):
    file = open(yaml_file, 'r', encoding="utf-8")         # 获取yaml文件数据
    data = yaml.load(file.read(), Loader=yaml.FullLoader) # 将yaml数据转化为字典
    file.close()
    if key_name is None:
        return data
    else:
        return data[key_name]
    

def write_yaml_data(yaml_file, data):
    file = open(yaml_file, 'w', encoding="utf-8")         # 获取yaml文件数据
    yaml.dump(data, file, allow_unicode=True)
    file.close()