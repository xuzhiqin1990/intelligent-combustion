
r'''
传入字典，将其转换为类。
若字典中的value也是字典，则递归转换为类。
'''
class Dict2Class(object):
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Dict2Class(v)
			else:
				self.__dict__[k] = v