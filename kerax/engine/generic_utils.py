def flatten(x):
    def _flatten(x, result=[]):
        for i in x:
            if isinstance(i, list):
                return _flatten(i, result)
            else:
                result.append(i)
        return result

    return _flatten(x, [])
