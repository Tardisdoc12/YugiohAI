################################################################################
# filename: BoardGame.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################

class GameBoard:
    def __init__(
        self,
        self_main_ids: list[int],  # les cartes que l'on possède en main
        self_field_ids: list[int],  # les cartes de notre côté du terrain
        self_graveyard_ids: list[int],  # les cartes qui sont dans notre cimetière,
        self_banish_ids: list[int],  # les cartes banies face visibles dans la zone de banissement,
        opp_fields_ids: list[int],  # les cartes du côté adverse visible
        opp_graveyard_ids: list[int],  # les cartes dans le cimetière adverse
        opp_banish_ids: list[int],  # les cartes adverse banies face recto
        self_banish_verso: int,  # les cartes banies face verso
        opp_banish_verso: int,  # les cartes adverses banies face verso
        opp_fields_verso_card: int,  # le nombre de carte du côté adverse mais face caché
        phase_id: int,  # la partie du tour qans laquelle on est
        lp: int,  # nos points de vie
        adv_lp: int,  # les points de notre adversaires
        opp_main: int,  # Le nombre de cartes que l'adversaire à en main
    ):
        self.self_main_ids = self_main_ids
        self.self_field_ids = self_field_ids
        self.self_graveyard_ids = self_graveyard_ids
        self.self_banish_ids = self_banish_ids
        self.opp_fields_ids = opp_fields_ids
        self.opp_graveyard_ids = opp_graveyard_ids
        self.opp_banish_ids = opp_banish_ids
        self.self_banish_verso = self_banish_verso
        self.opp_banish_verso = opp_banish_verso
        self.opp_fields_verso_card = opp_fields_verso_card
        self.phase_id = phase_id
        self.lp = lp
        self.adv_lp = adv_lp
        self.opp_main = opp_main

    def get_dict(self):
        return self.__dict__

    def append(self,zone : str, card_id : int | str)-> None:
        if card_id is None :
            print("Card_id cannot be None")
            return
        if zone not in self.__dict__.keys():
            print(f"Zone doesn't exist. Please select a zone between: {list(self.__dict__.keys())}")
            return
        if not isinstance(self.__dict__[zone], list):
            print("Selected Zone cannot contains cards Id")
            return
        if not isinstance(card_id,str) and not isinstance(card_id,int):
            print("Wrong type")
            return
        if isinstance(card_id,str):
            card_id = int(card_id)
        self.__dict__[zone].append(card_id)

    def extend(self,zone : str, card_ids : list[int | str])-> None:
        if card_ids is None :
            print("Card_ids cannot be None")
            return
        if zone not in self.__dict__.keys():
            print(f"Zone doesn't exist. Please select a zone between: {list(self.__dict__.keys())}")
            return
        if not isinstance(self.__dict__[zone], list):
            print("Selected Zone cannot contains cards Id")
            return
        if not isinstance(card_ids,list):
            print("Wrong type")
            return
        card_ids = list(map(int,card_ids))
        self.__dict__[zone].extend(card_ids)

    def __add__(self, zone : str) -> None:
        if self.get_dict().get(zone,None) is not None:
            self.__dict__[zone] += 1

    def reset(self):
        for key in self.get_dict().keys():
            if isinstance(self.get_dict()[key],list):
                self.__dict__[key] = []
            elif isinstance(self.get_dict()[key],int):
                self.__dict__[key] = 0

if __name__ == "__main__":
    toto = GameBoard([],[],[],[],[],[],[],0,0,0,0,8000,8000,5)
    toto.extend("self_main_ids", ["85685","421"])
    toto + "opp_banish_verso"
    toto + "self_banish_verso"
    print(toto.get_dict())
    toto.reset()
    print(toto.get_dict())


################################################################################
# End of File
################################################################################