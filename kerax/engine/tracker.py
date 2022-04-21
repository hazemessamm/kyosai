class Trackable:
    __name_uids = {}

    def __init__(self, name):
        self.name = Trackable.get_uid(name)

    @staticmethod
    def get_uid(string):
        Trackable.__name_uids[string] = Trackable.__name_uids.get(string, 0) + 1
        return f"{string}_{Trackable.__name_uids[string]}"

    @staticmethod
    def remove(name):
        uid = Trackable.__name_uids.get(name, None)
        if uid is not None:
            if uid == 1:
                del Trackable.__name_uids
            else:
                Trackable.__name_uids[name] -= 1

    @staticmethod
    def reset():
        Trackable.__name_uids.clear()
