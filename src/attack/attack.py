class abstract_attacker():
    def __init__(self):
        pass

    def attack(self):
        pass

class global_attacker(abstract_attacker):
    pass

class target_attacker(abstract_attacker):
    pass

class metattack(global_attacker):
    pass

class nettack(target_attacker):
    pass
