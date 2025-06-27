from typing import Any, Literal

def type_check_dictionary(dict: dict[Any, Any], key_type: type, value_type: type, item_scope: Literal["single", "all"]) -> bool:
    if item_scope == "single":
        item = dict.popitem()

        if isinstance(item[0], key_type) is False or isinstance(item[1], value_type) is False:
            return False
        else:
            return True
    else:
        for i, item in enumerate(list(dict.items())):
            if isinstance(item[0], key_type) is False:
                return False
            if isinstance(item[1], value_type) is False:
                return False
        return True