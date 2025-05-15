import torch
import torch.nn as nn


class GameStateEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

        # Pour encodage des scalaires (phase, LPs)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, embed_dim)  # phase, lp, adv_lp
        )

    def __transform_list_to_vector(self, list_to_transform: list, device):
        main_tensor = torch.tensor(list_to_transform, dtype=torch.long, device=device)
        if main_tensor.numel() > 0:
            main_embeds = self.embedding(main_tensor)
            main_vector = main_embeds.mean(dim=0)
        else:
            main_vector = torch.zeros(self.embed_dim, device=device)
        return main_vector

    def forward(
        self,
        self_main_ids: list[int],  # les cartes que l'on possède en main
        self_field_ids: list[int],  # les cartes de notre côté du terrain
        self_graveyard_ids: list[int],  # les cartes qui sont dans notre cimetière,
        self_banish_ids: list[
            int
        ],  # les cartes banies face visibles dans la zone de banissement,
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
        """
        Returns: tensor (embed_dim * 7,)
        """

        device = next(self.parameters()).device

        list_ids = [
            self_main_ids,
            self_field_ids,
            opp_fields_ids,
            self_graveyard_ids,
            self_banish_ids,
            opp_graveyard_ids,
            opp_banish_ids,
        ]

        vector_final = []
        for list_to_transform in list_ids:
            vector_final.append(
                self.__transform_list_to_vector(list_to_transform, device)
            )

        list_scalar = [
            phase_id,
            opp_main,
            lp,
            adv_lp,
            opp_fields_verso_card,
            self_banish_verso,
            opp_banish_verso,
        ]

        # Scalar features
        scalar_input = torch.tensor(
            list_scalar,
            dtype=torch.float32,
            device=device,
        )
        vector_final.append(self.scalar_mlp(scalar_input))

        # Concatène tout
        full_vector = torch.cat(vector_final, dim=0)  # (embed_dim * 8,)
        return full_vector
