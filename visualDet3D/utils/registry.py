import inspect

class Registry(object):
    
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def __getitem__(self, key):
        return self.module_dict[key]

    def register_module(self, module_class=None):

        if (not inspect.isclass(module_class)) and (not inspect.isfunction(module_class)):
            raise TypeError('module must be a class or function, but got {}'.format(type(module_class)))
        
        module_name = module_class.__name__

        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        
        # Save newly register module to dict 
        self._module_dict[module_name] = module_class
        
        return module_class

AUGMENTATION_DICT = Registry("augmentation")
