def are_all_members_same(param):
    if len(param) == 0:
        return True
    if param.count(param[0]) == len(param):
        return True
    else:
        return False


print are_all_members_same([1])