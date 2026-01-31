from .attack import Attacker
from .defend import Defender
from src.dataset.abstract_dataset import AbstractDataModule
class DefaultPipeline:
    def __init__(self, dataset_module:AbstractDataModule,defender:Defender,attacker:Attacker):
        self.dataset_module = dataset_module
        self.defender = defender
        self.attacker = attacker

    def attack_onely(self):

        pass

    def defend_after_attack(self):
        pass