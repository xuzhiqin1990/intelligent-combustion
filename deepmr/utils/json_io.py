import json
import re
from _ctypes import PyObj_FromPtr


# 使得输出的json文件列表不会换行
class NoIndent(object):
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr

def read_json_data(json_file_name):
    with open(json_file_name, 'r', encoding='utf8')as fp:
        args = json.load(fp)
    fp.close()
    return args

def write_json_data(json_file_name, args):
    with open(json_file_name, "w") as f:
        f.write(json.dumps(args, ensure_ascii=False, indent=4, separators=(',', ':')))
    f.close()

def save_simulation_to_json(datas, json_save_path, del_vector=True):
    '''
    将模拟结果保存为json文件
    删除vector key，并且模拟结果不换行
    '''
    # 删除字典中的vector key
    if 'vector' in datas.keys():
        if del_vector:
            del datas['vector']
        if not del_vector:
            datas['vector'] = NoIndent(datas['vector'].tolist())
    
    for indicator in datas.keys():
        if indicator == 'vector':
            continue
        for condition in datas[indicator].keys():
            datas[indicator][condition] = NoIndent(datas[indicator][condition])

    with open(json_save_path, 'w') as fw:
        json_data = json.dumps(datas, cls=MyEncoder, ensure_ascii=False, sort_keys=False, indent=2)
        fw.write(json_data)
        fw.write('\n')
    
