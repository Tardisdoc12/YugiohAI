################################################################################
# filename: mlp_yugioh.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################################################################################


class YuGiOhMLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_card_ids=1000, num_zones=5, num_actions=8
    ):
        super().__init__()

        # Tronc commun
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Tête pour l’id de la carte (sur tout le pool de cartes)
        self.card_head = nn.Linear(hidden_dim, num_card_ids)

        # Tête pour la zone (main, field, etc.)
        self.zone_head = nn.Linear(hidden_dim, num_zones)

        # Tête pour l’action (invoke, set, etc.)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.shared(x)

        card_logits = self.card_head(x)
        zone_logits = self.zone_head(x)
        action_logits = self.action_head(x)

        return card_logits, zone_logits, action_logits

################################################################################
# End of File
################################################################################